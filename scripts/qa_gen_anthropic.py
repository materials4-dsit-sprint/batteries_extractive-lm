"""
SQuAD QA Generator Pipeline — Anthropic Batch API + Haiku 4.5
=========================================================
TWO-PHASE WORKFLOW:
  Phase 1 — SUBMIT:   Convert all PDFs → submit one batch job → save batch_id
  Phase 2 — RETRIEVE: Poll until done → download results → merge into SQuAD dataset

This script is designed to run in Google Colab for easy access to Drive and GPU resources.
"""
!pip install anthropic
!pip install docling
import os, json, re, time
import anthropic
from docling.document_converter import DocumentConverter

ANTHROPIC_API_KEY = "insert-your-anthropic-api-key-here"
MODEL             = "claude-haiku-4-5-20251001"

# Target these specific property categories
BATTERY_PROPERTIES = [
    "specific capacity", "discharge capacity", "coulombic efficiency",
    "voltage plateau", "ionic conductivity", "energy density",
    "current density", "cycle life", "diffusion coefficient",
    "chemical composition", "lattice parameters", "particle size"
]

BATTERY_TOPICS = ["electrochemistry", "materials", "performance", "safety", "manufacturing"]


SYSTEM_PROMPT = """You are a Battery Informatics Extraction Engine.
Your goal is to build a high-precision dataset for materials property mining.
Focus exclusively on extracting numerical values, units, and chemical formulas.
Avoid descriptive or 'How-to' questions. Target 'What-is-the-Property-of-Material' patterns.
Return ONLY a raw JSON array."""




#
def build_informatics_prompt(text: str, filename: str) -> str:

    """
    This function constructs a detailed prompt for the Anthropic model, instructing it to extract specific types 
    of information from the provided paper text. 
    The prompt emphasizes the focus on numerical properties, chemical formulas, 
    and specific conditions related to battery materials, while also specifying 
    the desired question style and output format.

    Args : 
    text (str): The raw text of the battery science paper from which to extract QA pairs.
    filename (str): The name of the paper, used for logging and tracking purposes.

    Returns: 
    str: A formatted prompt string to be sent to the Anthropic model, containing instructions and the paper text.

    """

    return f"""Extract exactly 50 SQuAD-format QA pairs from the battery science paper below.

INFORMATICS EXTRACTION RULES:
1. FOCUS: Extract numerical properties (e.g., "170 mAh/g"), chemical formulas (e.g., "LiNi0.8Co0.1Mn0.1O2"), and specific conditions (e.g., "0.1 C").
2. QUESTION STYLE: Questions must be concise property-lookup style.
   - YES: "What is the specific capacity of [Material] at [Condition]?"
   - NO: "How does the author improve the stability of the electrolyte?"
3. TARGET PROPERTIES: {", ".join(BATTERY_PROPERTIES)}.
4. VERBATIM: The "context" must be a verbatim paragraph. The "answer" must be the EXACT substring (e.g., the number + unit).
5. IMPOSSIBLE QUESTIONS: Include 5 questions where a property is mentioned in the paragraph, but for a DIFFERENT material, and mark as "is_impossible": true.

Schema:
{{"context":"<verbatim_paragraph>","question":"<property_question>","answer":"<exact_value_or_formula>","answer_start":<int>,"topic":"<topic>","is_impossible":<bool>}}

PAPER TEXT:
{text[:180000]}"""



def parse_json_safe(content: str, label: str) -> list[dict]:


    """
    #This function takes the raw text response from Claude and attempts to extract a clean JSON array of QA pairs.
    #It first tries to parse the entire content as JSON, which works if Claude returns a clean JSON array without any markdown or extra text.

    #Robust JSON extractor — strips fences, handles truncation.

    Args: content (str): The raw text response from the Anthropic model, which may contain markdown formatting, extra text, or a JSON array.
          label (str): A label (e.g., filename) used for logging purposes to identify which response is being processed.

    Returns: list[dict]: A list of dictionaries representing the extracted QA pairs. Each dictionary should conform to the expected schema.
    If parsing fails or if the content does not contain a valid JSON array, an empty list
    is returned, and an error message is printed to the console.
    """
    content = re.sub(r'```(?:json)?\s*', '', content).strip().rstrip('`').strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    start = content.find('[')
    if start == -1:
        print(f"No JSON array in response for {label}")
        return []

    depth, end = 0, -1
    for i, ch in enumerate(content[start:], start):
        if ch == '[':   depth += 1
        elif ch == ']': depth -= 1
        if depth == 0:  end = i + 1; break

    if end == -1:
        print(f"Truncated JSON for {label}")
        return []

    try:
        data = json.loads(content[start:end])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"JSON parse failed for {label}: {e}")
        return []


def validate_and_fix(pairs: list[dict], filename: str) -> list[dict]:

    """
    This function ensures answer is a real substring of context. Fixes answer_start. Drops invalids.
    Args: pairs (list[dict]): A list of dictionaries representing the extracted QA pairs, where each dictionary should contain at least "context", "answer", and "answer_start" keys.
          filename (str): A label (e.g., filename) used for logging purposes to identify which set of QA pairs is being processed.
    Returns: list[dict]: A list of validated and corrected QA pairs. Each pair is checked to ensure that the "answer" is a substring of the "context". 
    If the "answer_start" index is incorrect, it is fixed. Pairs that fail validation (e.g., answer not in context) are dropped. 
    The function also logs the number of valid pairs, how many were fixed, and how many were dropped.

    """
    valid, fixed, dropped = [], 0, 0
    for pair in pairs:
        ctx = pair.get("context", "")
        ans = pair.get("answer",  "")
        if not ans or ans not in ctx:
            dropped += 1
            continue
        correct = ctx.find(ans)
        if pair.get("answer_start") != correct:
            pair["answer_start"] = correct
            fixed += 1
        if pair.get("topic") not in BATTERY_TOPICS:
            pair["topic"] = "performance"
        pair.setdefault("paper", filename)
        valid.append(pair)
    print(f"Valid: {len(valid)}  |  Fixed offsets: {fixed}  |  Dropped: {dropped}")
    return valid


def phase1_submit():
    """
    Converts all PDFs to markdown, builds one Batch API request per paper,
    submits everything in a single call, and saves the batch_id to Drive.
    Takes ~10-20 mins (just PDF conversion). No polling needed.
    """
    from google.colab import drive
    drive.mount('/content/drive')

    client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    converter  = DocumentConverter()
    dir_path   = '/content/drive/MyDrive/dsit/my_battery_papers'
    state_file = '/content/drive/MyDrive/dsit/batch_state.json'

    # Resume: don't re-submit if already done
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        if state.get("batch_id"):
            print(f"Already submitted. Batch ID: {state['batch_id']}")
            print(f"Run phase2_retrieve() to collect results.")
            return

    state    = {"batch_id": None, "id_map": {}}
    requests = []
    all_pdfs = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")])
    print(f"🚀 Converting {len(all_pdfs)} PDFs...\n")

    for i, filename in enumerate(all_pdfs, 1):
        # Sanitise filename into a valid custom_id (alphanum + _ -)
        custom_id = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)[:60]
        print(f"[{i}/{len(all_pdfs)}] {filename}")

        try:
            result = converter.convert(os.path.join(dir_path, filename))
            text   = result.document.export_to_markdown()
            print(f"  {len(text):,} chars")

            if len(text) < 500:
                print(f"Too short — skipping.")
                continue

            state["id_map"][custom_id] = filename
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model":       MODEL,
                    "max_tokens":  16000,
                    "temperature": 0,
                    "system":      SYSTEM_PROMPT,
                    "messages":    [{"role": "user", "content": build_informatics_prompt(text, filename)}]
                }
            })

        except Exception as e:
            print(f"Conversion error: {e}")

    print(f"Submitting batch of {len(requests)} requests to Anthropic...")

    batch = client.messages.batches.create(requests=requests)

    state["batch_id"] = batch.id
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"   Batch submitted successfully!")
    print(f"   Batch ID : {batch.id}")
    print(f"   Status   : {batch.processing_status}")
    print(f"   Papers   : {len(requests)}")
    print(f"   Anthropic will process this within 24 hours.")
    print(f"   Come back and run phase2_retrieve() to collect results.")
    print(f"   Your batch_id is saved to: {state_file}")




def phase2_retrieve():
    """
    Polls the batch every 60s until complete, then:
    - Downloads all results
    - Validates + fixes each paper's QA pairs
    - Saves per-paper JSONs to output_squad/
    - Merges everything into a single squad_dataset.json
    """
    from google.colab import drive
    drive.mount('/content/drive')

    client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    state_file  = '/content/drive/MyDrive/dsit/batch_state.json'
    output_base = '/content/drive/MyDrive/dsit/output_squad_props'
    os.makedirs(output_base, exist_ok=True)

    if not os.path.exists(state_file):
        print("No state file — run phase1_submit() first.")
        return

    with open(state_file) as f:
        state = json.load(f)

    batch_id = state.get("batch_id")
    id_map   = state.get("id_map", {})

    if not batch_id:
        print("No batch_id found — run phase1_submit() first.")
        return

    print(f"Polling batch: {batch_id}\n")

    # --- Poll until ended ---
    while True:
        batch  = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  [{time.strftime('%H:%M:%S')}] {status} — "
              f"processing: {counts.processing}  "
              f"succeeded: {counts.succeeded}  "
              f"errored: {counts.errored}")

        if status == "ended":
            break
        time.sleep(60)

    print(f"Batch complete! Downloading results...\n")

    # --- Stream all results into a dict ---
    results_by_id = {}
    for result in client.messages.batches.results(batch_id):
        results_by_id[result.custom_id] = result
    print(f" Downloaded {len(results_by_id)} results")

    # --- Process each result ---
    total_saved = 0
    for custom_id, filename in id_map.items():
        out_file = os.path.join(output_base, f"{filename}.json")

        if os.path.exists(out_file):
            print(f"\n  Skip (exists): {filename}")
            continue

        result = results_by_id.get(custom_id)
        if not result:
            print(f"\n  Missing result for {filename}")
            continue

        print(f"\n {filename}")

        if result.result.type == "error":
            print(f"  Error: {result.result.error}")
            continue

        msg     = result.result.message
        content = msg.content[0].text.strip()

        if msg.stop_reason == "max_tokens":
            print(f" Hit max_tokens — result may be partial")

        pairs     = parse_json_safe(content, custom_id)
        validated = validate_and_fix(pairs, filename)

        if validated:
            with open(out_file, 'w') as f:
                json.dump(validated, f, indent=2)
            print(f" Saved {len(validated)} pairs")
            total_saved += len(validated)
        else:
            print(f" 0 valid pairs saved")

    print(f"Total QA pairs saved across all papers: {total_saved}")

    # --- Final merge into one SQuAD file ---
    squad_path = os.path.join(output_base, "squad_dataset_props.json")
    merge_to_squad_format(output_base, squad_path)



def merge_to_squad_format(output_base: str, squad_out: str):
    """Merges all per-paper JSONs into a single HuggingFace SQuAD-compatible file.
    
    Args : output_base (str): The directory where individual per-paper JSON files are stored.
              squad_out (str): The file path where the merged SQuAD dataset will be saved
    
    Returns: None. The function writes the merged dataset to the specified squad_out path.
    The function reads all JSON files in the output_base directory, groups QA pairs by their context, 
    and constructs a SQuAD-format dataset. Each unique context becomes a "paragraph" with associated
    QA pairs. The final dataset is saved as a JSON file at the squad_out location.
    
    """
    all_files = sorted([
        f for f in os.listdir(output_base)
        if f.endswith(".json") and f != "squad_dataset_props.json"
    ])

    squad    = {"version": "battery-squad-1.0", "data": []}
    total_qa = 0

    for fname in all_files:
        with open(os.path.join(output_base, fname)) as f:
            pairs = json.load(f)
        if not pairs:
            continue

        ctx_map: dict[str, list] = {}
        for pair in pairs:
            ctx_map.setdefault(pair.get("context", ""), []).append(pair)

        paragraphs = []
        for ctx, ctx_pairs in ctx_map.items():
            qas = [{
                "id":            f"{fname}_{j}",
                "question":      p.get("question", ""),
                "answers":       [{"text": p.get("answer", ""), "answer_start": p.get("answer_start", 0)}],
                "topic":         p.get("topic", "performance"),
                "is_impossible": False
            } for j, p in enumerate(ctx_pairs)]
            paragraphs.append({"context": ctx, "qas": qas})

        squad["data"].append({"title": fname.replace(".json", ""), "paragraphs": paragraphs})
        total_qa += len(pairs)

    with open(squad_out, 'w') as f:
        json.dump(squad, f, indent=2)

    print(f"\n SQuAD dataset saved → {squad_out}")
    print(f"   Papers:   {len(squad['data'])}")
    print(f"   QA pairs: {total_qa}")
    print(f"\nLoad in HuggingFace with:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('json', data_files='{squad_out}')")




if __name__ == "__main__":
    #── STEP 1: Run this first, then close Colab if you want ──
    phase1_submit()

    # ── STEP 2: Run this after batch completes (check email or poll) ──
    #phase2_retrieve()
