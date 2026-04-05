"""
ToolReliBench: REAL API Runner
===============================
Makes actual LLM API calls to:
  - OpenAI    (GPT-4o-mini)       key starts with: sk-
  - Google    (Gemini 1.5 Flash)  key starts with: AIza
  - xAI Grok  (grok-beta)         key starts with: xai-

Usage:
    python run_real_experiment.py \
        --openai-key  sk-...   \
        --gemini-key  AIza...  \
        --grok-key    xai-...  \
        --runs-per-task 3      \
        --output-dir  ./real_results

Cost estimate (3 runs x 20 tasks x single arch):
    GPT-4o-mini   ~$0.50-1.00
    Gemini Flash  ~$0.20-0.50
    Grok beta     ~$1.00-2.00
"""

import os, sys, json, time, re, argparse, logging, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import urllib.request
import urllib.error

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.metrics import MetricAggregator, compute_cdi, compute_svr, compute_ric
from src.evaluation_pipeline import run_full_analysis
from src.visualizations import save_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RealRunner")

# ─────────────────────────────────────────────────────────────────────────────
# Tasks (20 tasks: 10 short + 10 long)
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [
    # SHORT (5-10 steps expected)
    {"task_id":"short_000","description":"Calculate the compound interest on $10,000 invested at 5% annual rate for 10 years. Show each step of the calculation.","expected_steps":5,"category":"calculation","difficulty":"short"},
    {"task_id":"short_001","description":"Search your knowledge for Tokyo's population. Then verify whether it is the largest city in the world by comparing with Delhi and Shanghai populations.","expected_steps":6,"category":"research","difficulty":"short"},
    {"task_id":"short_002","description":"The S&P 500 is at 4783. Calculate: (a) what $5,000 would buy in index units, (b) value after 7% annual growth for 5 years.","expected_steps":7,"category":"finance","difficulty":"short"},
    {"task_id":"short_003","description":"List the capital of France, Germany, and Japan. Then calculate the approximate straight-line distance between Paris and Berlin (use 878 km as the value) and convert to miles.","expected_steps":5,"category":"geography","difficulty":"short"},
    {"task_id":"short_004","description":"Explain what Python programming language is. Confirm whether it is a high-level or low-level language and give 2 specific reasons why.","expected_steps":4,"category":"verification","difficulty":"short"},
    {"task_id":"short_005","description":"US GDP in 2023 was $27.36 trillion. The population was 335 million. Calculate GDP per capita and compare it to the world average of $13,000.","expected_steps":6,"category":"economics","difficulty":"short"},
    {"task_id":"short_006","description":"World median age is 30.9 years. World population is 8 billion. If people under 30 are roughly 50% of the population, compute the adult population and the youth population.","expected_steps":5,"category":"demographics","difficulty":"short"},
    {"task_id":"short_007","description":"NASDAQ is at 15,074. S&P 500 is at 4,783. Calculate the ratio. Interpret what it means when this ratio is above 3 vs below 3 historically.","expected_steps":6,"category":"finance","difficulty":"short"},
    {"task_id":"short_008","description":"Define machine learning. Then categorise these 3 algorithms: linear regression, k-means clustering, Q-learning — into supervised, unsupervised, or reinforcement learning.","expected_steps":6,"category":"verification","difficulty":"short"},
    {"task_id":"short_009","description":"The nominal interest rate is 5.25%. Inflation is 3.4%. Calculate the real interest rate using the Fisher equation. Then state whether this is positive or negative real returns.","expected_steps":5,"category":"calculation","difficulty":"short"},

    # LONG (15-28 steps expected)
    {"task_id":"long_000","description":"""Conduct a comprehensive financial analysis across 8 steps:
1. State the current S&P 500 value (use 4783) and NASDAQ (use 15074). Calculate their ratio.
2. State US GDP for 2023 ($27.36 trillion). Calculate GDP per capita (population: 335M).
3. State the 2023 inflation rate (3.4%). Calculate real GDP growth if nominal was 2.5%.
4. State unemployment rate (3.7%). Assess whether this is above or below the natural rate (4.5%).
5. Calculate the real interest rate: nominal 5.25% minus inflation 3.4%.
6. Assess the yield curve: 10Y Treasury at 4.5%, 2Y at 4.9%. Is it inverted? What does that signal?
7. Cross-verify all 6 indicators for internal consistency. Flag any contradictions.
8. Synthesise into an overall economic outlook: expansion, contraction, or stagflation?""","expected_steps":20,"category":"financial_analysis","difficulty":"long"},

    {"task_id":"long_001","description":"""Perform multi-step global demographic research:
1. State world population (8 billion). State top 5 countries by population.
2. Calculate what percentage of world population the top 5 represent.
3. State global median age (30.9 years) and urbanisation rate (57%).
4. Calculate urban vs rural population split.
5. State global population growth rate (0.9% per year).
6. Project world population for 2030 and 2050 using compound growth.
7. Identify the 3 fastest-growing regions (Sub-Saharan Africa, South Asia, Southeast Asia).
8. Verify: does the top-5 country percentage + rest-of-world = 100%? Fix if not.
9. State the demographic transition model stage for each top-5 country.
10. Summarise: what are the 3 biggest demographic challenges for 2050?""","expected_steps":22,"category":"demographic_research","difficulty":"long"},

    {"task_id":"long_002","description":"""Complete a chained economic data analysis:
1. Retrieve inflation rate (3.4%) and explain what CPI measures.
2. State Federal Reserve's current rate (5.25-5.5% range). Why is it this high?
3. Calculate real interest rate. Is monetary policy restrictive or accommodative?
4. State unemployment (3.7%). Apply Okun's Law: if GDP grows 1%, unemployment falls ~0.5%.
5. If GDP growth is 2.5%, what does Okun's Law predict for unemployment change?
6. State the Misery Index = inflation + unemployment. Interpret the result.
7. Compare to 1980 Misery Index (~20%) and 2019 (~6%). Where are we now?
8. Cross-reference all data points: do they tell a consistent story?
9. Identify the single biggest risk to the current economic outlook.
10. Provide a final economic health score from 1-10 with justification.""","expected_steps":20,"category":"economic_analysis","difficulty":"long"},

    {"task_id":"long_003","description":"""Execute a recursive verification chain on AI/ML:
1. Define Artificial Intelligence. Give the year of the Dartmouth Conference (1956).
2. Define Machine Learning. State when deep learning took off (2012, ImageNet).
3. Define Deep Learning. Explain how it differs from classical ML.
4. Verify: Is deep learning a subset of ML? Is ML a subset of AI? Draw the hierarchy.
5. State the 3 main types of ML (supervised, unsupervised, reinforcement). Give one example each.
6. Define a neural network. State how many layers makes it 'deep' (>1 hidden layer).
7. Verify the claim: 'GPT-4 is a large language model based on the transformer architecture.' True or false and why?
8. Cross-reference: does your definition of LLM align with your definition of deep learning?
9. Identify one claim you made above that you are LEAST confident about. State why.
10. Provide a verified, corrected final summary of the AI/ML/DL hierarchy.""","expected_steps":24,"category":"verification_chain","difficulty":"long"},

    {"task_id":"long_004","description":"""Perform a tool composition analysis on programming and AI adoption:
1. State the top 5 programming languages by popularity (TIOBE or similar ranking).
2. State Python's rank and why it dominates data science.
3. State AI adoption rate in enterprises (estimate: ~35% in 2023).
4. Calculate: if AI adoption grows 20% per year, what is the rate in 2027?
5. State the number of data science / ML job postings growth (estimate: 35% YoY).
6. Calculate correlation insight: does Python popularity track AI adoption? Explain.
7. State the top 3 Python libraries for ML (scikit-learn, TensorFlow, PyTorch).
8. Verify: Is PyTorch or TensorFlow more popular in research papers? State your evidence.
9. Cross-reference all claims: are the job market trends consistent with language popularity trends?
10. Identify any data point above where you had to estimate. Flag uncertainty.
11. Final synthesis: recommend one programming language for someone entering AI in 2025. Justify.""","expected_steps":26,"category":"tool_composition","difficulty":"long"},

    {"task_id":"long_005","description":"""Multi-hop research on climate and energy economics:
1. State global CO2 emissions in 2023 (approximately 37 billion tonnes).
2. State the Paris Agreement target: limit warming to 1.5°C above pre-industrial.
3. Calculate the remaining carbon budget (approximately 400 Gt CO2 at current rates, ~10 years).
4. State global renewable energy share (~30% of electricity in 2023).
5. Calculate: at 30% renewable and growing 3 percentage points per year, when do we hit 100%?
6. State the cost of solar per kWh in 2023 (~$0.04) vs 2010 (~$0.38). Calculate cost reduction %.
7. State current global carbon price range ($10-$130/tonne depending on region).
8. Calculate economic impact: at $50/tonne × 37Gt = total carbon cost globally.
9. Cross-verify: is the math in steps 5 and 8 internally consistent with step 3?
10. Identify the single most important lever to meet the 1.5°C target.""","expected_steps":22,"category":"climate_economics","difficulty":"long"},

    {"task_id":"long_006","description":"""Planning task — design an optimal 5-city tech conference tour:
Cities to consider: San Francisco, London, Singapore, Tokyo, Berlin.
1. State the purpose: maximise attendee reach across North America, Europe, Asia.
2. List population of tech workers in each city (estimate: SF 200K, London 300K, Singapore 100K, Tokyo 250K, Berlin 150K).
3. Calculate total addressable audience.
4. Propose an optimal order minimising total flight distance. State distances between cities.
5. Calculate total travel distance for your proposed route.
6. Compare to the reverse route. Which is shorter?
7. Factor in timezones: identify which city pair has the worst jetlag (time difference > 8 hours).
8. Budget constraint: $50,000 for venue rental. Rank cities by estimated venue cost (SF highest, Berlin lowest).
9. Identify which city to cut if budget must drop to $40,000.
10. Final plan: 5-city order, total distance, total estimated cost, dates avoiding major holidays.""","expected_steps":24,"category":"planning","difficulty":"long"},

    {"task_id":"long_007","description":"""Cross-reference synthesis — US economic data from multiple sources:
1. State US GDP from IMF ($27.36T), World Bank ($25.46T adj), and BEA ($27.36T). Why do they differ?
2. State US inflation from CPI (3.4%) vs PCE (2.6%). Explain why the Fed prefers PCE.
3. State unemployment from BLS U-3 (3.7%) vs U-6 (7.0%). What does U-6 include that U-3 doesn't?
4. Cross-reference GDP sources: which is most authoritative for comparing across countries?
5. Cross-reference inflation measures: which is higher and by how much? Calculate the gap.
6. Identify one implication of using U-6 vs U-3 for policy decisions.
7. If you had to report ONE number for US economic health to a foreign investor, which GDP figure would you use and why?
8. Synthesise: do these different measurement approaches give a consistent picture, or are there genuine contradictions?""","expected_steps":22,"category":"cross_reference","difficulty":"long"},

    {"task_id":"long_008","description":"""Forecast AI adoption using S-curve (logistic) analysis:
1. State AI adoption in 2020 (10%), 2021 (15%), 2022 (25%), 2023 (35%).
2. Fit a simple logistic growth: carrying capacity K=90%, current rate suggests midpoint ~2026.
3. Calculate predicted adoption for 2025, 2026, 2027, 2028 using logistic formula.
4. Compare S-curve adoption to historical tech: internet (reached 50% in ~7 years from mass launch), mobile (5 years). Is AI faster or slower?
5. State the main barrier slowing AI adoption (skills gap, cost, trust — pick the most evidence-based).
6. If the skills gap reduces adoption by 15%, recalculate 2027 prediction.
7. State 3 industries with highest AI adoption rates (finance, healthcare, retail).
8. Calculate: if finance is at 60% adoption and average is 35%, what is the gap in percentage points?
9. Cross-verify your logistic model: does step 6 adjustment stay within the bounds of the model?
10. Final forecast: in what year does AI adoption cross 70% globally? State your confidence level.""","expected_steps":24,"category":"forecasting","difficulty":"long"},

    {"task_id":"long_009","description":"""Error recovery task — identify and correct mistakes in economic data:
You are given the following dataset. Some values are wrong. Find and fix them.
Data: GDP_USA=$2.7T, Population_World=800M, Inflation_2023=34%, UnemploymentUSA=37%, SolarCost_2023=$0.40/kWh, CO2_emissions=3.7Gt, SP500=478, PythonRank=7th, MedianAge=90years, RenewableShare=3%.

1. Check each value against your knowledge. Flag which are wrong and why.
2. Provide the correct value for each flagged error.
3. Calculate the magnitude of each error (e.g. GDP is off by 10x).
4. Identify the 3 most dangerous errors (ones most likely to cause downstream miscalculation).
5. Re-run the corrected dataset: calculate GDP per capita, real interest rate (assume nominal 5.25%), misery index.
6. Verify your corrections are self-consistent (e.g. does corrected CO2 + corrected renewable share make sense?).
7. Identify any value you could NOT verify from your training knowledge and explain why.
8. Final output: corrected dataset with confidence rating (high/medium/low) for each value.""","expected_steps":20,"category":"error_recovery","difficulty":"long"},
]

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS  (passed to models)
# ─────────────────────────────────────────────────────────────────────────────
TOOLS_OPENAI = [
    {"type":"function","function":{
        "name":"calculator",
        "description":"Evaluate a mathematical expression and return the numeric result.",
        "parameters":{"type":"object","properties":{"expression":{"type":"string","description":"A valid Python math expression e.g. '10000 * (1.05**10)'"}},"required":["expression"]}
    }},
    {"type":"function","function":{
        "name":"search",
        "description":"Retrieve factual information from your knowledge base about a topic.",
        "parameters":{"type":"object","properties":{"query":{"type":"string","description":"The information to look up"}},"required":["query"]}
    }},
    {"type":"function","function":{
        "name":"verify",
        "description":"Verify whether a stated fact is correct by cross-checking it.",
        "parameters":{"type":"object","properties":{"claim":{"type":"string","description":"The claim to verify"}},"required":["claim"]}
    }},
]

TOOLS_ANTHROPIC = [
    {"name":"calculator","description":"Evaluate a mathematical expression.","input_schema":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}},
    {"name":"search","description":"Retrieve factual information.","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},
    {"name":"verify","description":"Verify a claim.","input_schema":{"type":"object","properties":{"claim":{"type":"string"}},"required":["claim"]}},
]

# Tool execution (local)
def _execute_tool(name: str, params: dict) -> str:
    if name == "calculator":
        expr = params.get("expression","")
        try:
            safe = {"abs":abs,"max":max,"min":min,"pow":pow,"round":round,"sum":sum}
            result = eval(expr, {"__builtins__":{}}, safe)
            return f"Result: {result}"
        except Exception as e:
            return f"CalculatorError: {e}"
    elif name in ("search","verify"):
        q = params.get("query") or params.get("claim","")
        return f"[Knowledge base result for: {q}] — please use your training knowledge to answer this."
    return f"Unknown tool: {name}"

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper (stdlib only — no requests needed)
# ─────────────────────────────────────────────────────────────────────────────
def _http_post(url: str, headers: dict, body: dict, timeout: int = 90) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body_text[:400]}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient:
    """GPT-4o-mini via OpenAI API."""
    def __init__(self, api_key: str):
        self.key = api_key
        self.model = "gpt-4o-mini"
        self.url   = "https://api.openai.com/v1/chat/completions"

    def complete(self, messages: list, tools: list = None) -> dict:
        headers = {"Content-Type":"application/json","Authorization":f"Bearer {self.key}"}
        body: dict = {"model":self.model,"messages":messages,"temperature":0.7,"max_tokens":800}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        for attempt in range(3):
            try:
                resp = _http_post(self.url, headers, body)
                choice  = resp["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                usage   = resp.get("usage",{})

                # Extract tool calls
                raw_tcs = message.get("tool_calls") or []
                tool_calls = []
                for tc in raw_tcs:
                    fn = tc.get("function",{})
                    try:
                        args = json.loads(fn.get("arguments","{}"))
                    except:
                        args = {}
                    tool_calls.append({"name":fn.get("name",""), "parameters":args})

                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "tokens": {"prompt":usage.get("prompt_tokens",0),
                               "completion":usage.get("completion_tokens",0),
                               "total":usage.get("total_tokens",0)},
                    "finish_reason": choice.get("finish_reason",""),
                }
            except Exception as e:
                if "401" in str(e):
                    raise RuntimeError(f"INVALID_API_KEY for {self.model} — get a new key") from e
                if attempt == 2: raise
                wait = 2 ** attempt
                log.warning(f"OpenAI retry {attempt+1}/3 after {wait}s: {e}")
                time.sleep(wait)


class GeminiClient:
    """Gemini 1.5 Flash via Google Generative Language API."""
    def __init__(self, api_key: str):
        self.key   = api_key
        self.model = "gemini-1.5-flash"

    def complete(self, messages: list, tools: list = None) -> dict:
        # Convert OpenAI-style messages → Gemini contents
        contents = []
        system_text = ""
        for m in messages:
            role = m["role"]
            text = m.get("content","") or ""
            if role == "system":
                system_text = text
            elif role == "user":
                contents.append({"role":"user","parts":[{"text":text}]})
            elif role == "assistant":
                contents.append({"role":"model","parts":[{"text":text}]})
            elif role == "tool":
                contents.append({"role":"user","parts":[{"text":f"[Tool result]: {text}"}]})

        if not contents:
            contents.append({"role":"user","parts":[{"text":"Begin."}]})

        body: dict = {
            "contents": contents,
            "generationConfig": {"temperature":0.7,"maxOutputTokens":800},
        }
        if system_text:
            body["systemInstruction"] = {"parts":[{"text":system_text}]}

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.key}")
        headers = {"Content-Type":"application/json"}

        for attempt in range(3):
            try:
                resp = _http_post(url, headers, body)
                candidate = resp.get("candidates",[{}])[0]
                parts = candidate.get("content",{}).get("parts",[])
                content = " ".join(p.get("text","") for p in parts if "text" in p)
                usage = resp.get("usageMetadata",{})
                return {
                    "content": content,
                    "tool_calls": [],   # Gemini Flash function calling needs SDK; skip for now
                    "tokens": {
                        "prompt":     usage.get("promptTokenCount",0),
                        "completion": usage.get("candidatesTokenCount",0),
                        "total":      usage.get("totalTokenCount",0),
                    },
                    "finish_reason": candidate.get("finishReason",""),
                }
            except Exception as e:
                if attempt == 2: raise
                wait = 2 ** attempt
                log.warning(f"Gemini retry {attempt+1}/3 after {wait}s: {e}")
                time.sleep(wait)


class GrokClient:
    """Grok (grok-beta) via xAI API — OpenAI-compatible endpoint."""
    def __init__(self, api_key: str):
        self.key   = api_key
        self.model = "grok-beta"
        self.url   = "https://api.x.ai/v1/chat/completions"

    def complete(self, messages: list, tools: list = None) -> dict:
        headers = {"Content-Type":"application/json","Authorization":f"Bearer {self.key}"}
        body: dict = {"model":self.model,"messages":messages,"temperature":0.7,"max_tokens":800}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        for attempt in range(3):
            try:
                resp = _http_post(self.url, headers, body)
                choice  = resp["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                usage   = resp.get("usage",{})

                raw_tcs = message.get("tool_calls") or []
                tool_calls = []
                for tc in raw_tcs:
                    fn = tc.get("function",{})
                    try:
                        args = json.loads(fn.get("arguments","{}"))
                    except:
                        args = {}
                    tool_calls.append({"name":fn.get("name",""), "parameters":args})

                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "tokens": {"prompt":usage.get("prompt_tokens",0),
                               "completion":usage.get("completion_tokens",0),
                               "total":usage.get("total_tokens",0)},
                    "finish_reason": choice.get("finish_reason",""),
                }
            except Exception as e:
                if attempt == 2: raise
                wait = 2 ** attempt
                log.warning(f"Grok retry {attempt+1}/3 after {wait}s: {e}")
                time.sleep(wait)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT — executes one task, returns a complete trace
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise analytical agent solving multi-step tasks.
You have access to these tools:
- calculator(expression): evaluates math expressions
- search(query): retrieves factual information
- verify(claim): checks whether a claim is correct

Rules:
1. Work step by step — number each step clearly.
2. When you perform a calculation, USE the calculator tool.
3. When you look something up, USE the search tool.
4. When you verify a fact, USE the verify tool — do NOT just say "verified" without calling it.
5. After every 5 steps, write a one-sentence reflection on your progress.
6. End with "FINAL ANSWER:" followed by a clear summary.
"""

def run_agent(client, task: dict, max_steps: int = 30, use_tools: bool = True) -> dict:
    """Run the agent on one task. Returns a complete trace dict."""
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": task["description"]},
    ]
    tools = TOOLS_OPENAI if use_tools and not isinstance(client, GeminiClient) else None

    steps = []
    total_tokens = {"prompt":0,"completion":0,"total":0}

    for step_num in range(1, max_steps + 1):
        t0 = time.time()
        try:
            resp = client.complete(messages, tools)
        except Exception as e:
            log.error(f"  Step {step_num} API error: {e}")
            break

        latency_ms = (time.time() - t0) * 1000
        content    = resp.get("content","") or ""
        tool_calls = resp.get("tool_calls",[])
        tok        = resp.get("tokens",{})

        for k in total_tokens:
            total_tokens[k] += tok.get(k,0)

        # Execute tool calls
        tool_outputs = []
        for tc in tool_calls:
            result = _execute_tool(tc["name"], tc.get("parameters",{}))
            tool_outputs.append(result)

        # Reflection every 5 steps
        reflection = None
        if step_num % 5 == 0 and content:
            reflection_match = re.search(
                r'(reflect|progress|on track|summary so far)[^\n]*', content, re.I)
            reflection = reflection_match.group(0) if reflection_match else None

        steps.append({
            "step_number":   step_num,
            "reasoning":     content,
            "tool_calls":    [{"name":tc["name"],"parameters":tc.get("parameters",{})}
                              for tc in tool_calls],
            "tool_outputs":  tool_outputs,
            "memory_state":  {"step":step_num,"tokens_so_far":total_tokens["total"]},
            "reflection":    reflection,
            "token_usage":   tok,
            "latency_ms":    round(latency_ms,1),
            "timestamp":     datetime.utcnow().isoformat(),
        })

        # Append assistant turn
        messages.append({"role":"assistant","content": content or "[tool call]"})

        # Append tool results
        for tc, output in zip(tool_calls, tool_outputs):
            messages.append({"role":"user","content": f"Tool '{tc['name']}' returned: {output}"})

        # Termination check
        finish = resp.get("finish_reason","")
        if finish == "stop" or "FINAL ANSWER" in content.upper():
            break
        if finish == "length":
            log.warning(f"  Step {step_num}: max tokens hit, continuing…")

        # Rate limit buffer
        time.sleep(0.3)

    # Determine success
    last_text = steps[-1]["reasoning"].upper() if steps else ""
    success   = "FINAL ANSWER" in last_text or "CONCLUSION" in last_text or len(steps) >= 4

    # Primary failure type detection
    failure_type = None
    if not success:
        failure_type = "INCOMPLETE"
    else:
        all_text = " ".join(s["reasoning"] for s in steps).lower()
        if re.search(r'\bverif(ied|y)\b|\bconfirm(ed)?\b', all_text):
            no_tool_steps = [s for s in steps
                             if re.search(r'\bverif(ied|y)\b|\bconfirm(ed)?\b', s["reasoning"], re.I)
                             and not s["tool_calls"]]
            if len(no_tool_steps) > 2:
                failure_type = "SILENT_VERIFICATION"

    return {
        "task_id":          task["task_id"],
        "model":            getattr(client, "model", "unknown"),
        "architecture":     "single",
        "task_description": task["description"],
        "task_category":    task.get("category",""),
        "task_difficulty":  task.get("difficulty",""),
        "steps":            steps,
        "success":          success,
        "failure_type":     failure_type,
        "total_tokens":     total_tokens["total"],
        "total_cost":       _estimate_cost(getattr(client,"model",""), total_tokens),
        "total_latency_ms": sum(s["latency_ms"] for s in steps),
        "metadata": {
            "architecture":    "single",
            "n_steps":         len(steps),
            "expected_steps":  task.get("expected_steps",0),
            "tool_calls_total":sum(len(s["tool_calls"]) for s in steps),
        }
    }


def _estimate_cost(model: str, tokens: dict) -> float:
    pricing = {
        "gpt-4o-mini":      {"in":0.00015, "out":0.0006},
        "gemini-1.5-flash": {"in":0.000075,"out":0.0003},
        "grok-beta":        {"in":0.005,   "out":0.015},
    }
    p = pricing.get(model, {"in":0.0001,"out":0.0004})
    return (tokens.get("prompt",0)/1000)*p["in"] + (tokens.get("completion",0)/1000)*p["out"]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(clients: dict, runs_per_task: int, output_dir: Path):
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    all_traces: dict = {name: [] for name in clients}
    total_cost  = 0.0
    completed   = 0
    total_runs  = len(clients) * len(TASKS) * runs_per_task

    log.info("=" * 65)
    log.info("ToolReliBench — REAL API Experiment")
    log.info(f"  Models:        {list(clients.keys())}")
    log.info(f"  Tasks:         {len(TASKS)}")
    log.info(f"  Runs per task: {runs_per_task}")
    log.info(f"  Total API calls: {total_runs}")
    log.info("=" * 65)

    for model_name, client in clients.items():
        model_dir = traces_dir / model_name.replace("/","_").replace(":","_")
        model_dir.mkdir(exist_ok=True)
        run_idx = 0
        model_failed = False

        log.info(f"\n▶  Model: {model_name}")

        for task in TASKS:
            if model_failed:
                break
            for run in range(runs_per_task):
                log.info(f"   [{completed+1}/{total_runs}] {task['task_id']} run {run+1}/{runs_per_task}")
                try:
                    trace = run_agent(client, task)
                except RuntimeError as e:
                    if "INVALID_API_KEY" in str(e):
                        log.error(f"\n❌  {model_name}: Invalid API key — skipping this model entirely.")
                        log.error(f"   {e}")
                        model_failed = True
                        completed += (len(TASKS) - TASKS.index(task)) * runs_per_task - run
                        break
                    log.error(f"   FAILED: {e}")
                    run_idx += 1
                    completed += 1
                    continue
                except Exception as e:
                    log.error(f"   FAILED: {e}\n{traceback.format_exc()}")
                    run_idx += 1
                    completed += 1
                    continue

                trace_path = model_dir / f"run_{run_idx:04d}.json"
                trace_path.write_text(json.dumps(trace, indent=2, default=str))

                all_traces[model_name].append(trace)
                total_cost += trace.get("total_cost", 0)
                run_idx    += 1
                completed  += 1

                cdi = compute_cdi(trace)["cdi"]
                svr = compute_svr(trace)["svr"]
                ric = compute_ric(trace)["ric"]
                status = "✓" if trace["success"] else "✗"
                log.info(f"   {status} steps={trace['metadata']['n_steps']:2d}  "
                         f"CDI={cdi:.3f}  SVR={svr:.3f}  RIC={ric:.3f}  "
                         f"cost=${trace['total_cost']:.4f}")

                # Small delay between runs — be a good API citizen
                time.sleep(1.0)

    log.info(f"\n✅  Done. Total estimated cost: ${total_cost:.3f}")
    return all_traces, total_cost


def generate_report(all_traces: dict, output_dir: Path):
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    log.info("\nRunning statistical analysis…")
    results = run_full_analysis(all_traces)

    # Save JSON
    (analysis_dir / "metrics.json").write_text(
        json.dumps(results["model_summaries"], indent=2, default=str))

    # Save CSVs
    for key, fname in [
        ("statistical_tests",  "statistical_tests.csv"),
        ("reflection_analysis","reflection_analysis.csv"),
        ("delegation_threshold","delegation_threshold.csv"),
    ]:
        df = results.get(key)
        if df is not None and not df.empty:
            df.to_csv(analysis_dir / fname, index=False)

    log.info("Generating plots…")
    save_all_plots(results, str(analysis_dir))

    # Markdown report
    sums = results["model_summaries"]
    lines = [
        "# ToolReliBench — REAL API Results\n",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total Trajectories:** {sum(v['n_trajectories'] for v in sums.values())}",
        f"**Models:** {', '.join(sums.keys())}\n",
        "## Results\n",
        "| Model | Success % | CDI (↓) | SVR (↓) | RIC (↓) | Tool Hall. (↓) |",
        "|-------|-----------|---------|---------|---------|----------------|",
    ]
    for m, s in sums.items():
        lines.append(
            f"| {m} | {s['success_rate']*100:.1f}% "
            f"| {s['cdi']['mean']:.3f} ±{s['cdi']['std']:.3f} "
            f"| {s['svr']['mean']:.3f} ±{s['svr']['std']:.3f} "
            f"| {s['ric']['mean']:.3f} ±{s['ric']['std']:.3f} "
            f"| {s['tool_hallucination_rate']['mean']:.3f} |"
        )

    st = results.get("statistical_tests")
    if st is not None and not st.empty:
        lines += ["\n## Statistical Comparisons\n", st.to_markdown(index=False)]

    ref = results.get("reflection_analysis")
    if ref is not None and not ref.empty:
        lines += ["\n## Self-Reflection Paradox\n", ref.to_markdown(index=False)]

    lines.append("\n---\n*Real API runs — ToolReliBench*")
    (analysis_dir / "research_report.md").write_text("\n".join(lines))

    # Summary to stdout
    print("\n" + "=" * 65)
    print("FINAL RESULTS (real API runs)")
    print("=" * 65)
    for m, s in sums.items():
        print(f"\n{m}:")
        print(f"  Success Rate : {s['success_rate']*100:.1f}%")
        print(f"  CDI          : {s['cdi']['mean']:.3f} ± {s['cdi']['std']:.3f}")
        print(f"  SVR          : {s['svr']['mean']:.3f} ± {s['svr']['std']:.3f}")
        print(f"  RIC          : {s['ric']['mean']:.3f} ± {s['ric']['std']:.3f}")
        print(f"  Tool Hall.   : {s['tool_hallucination_rate']['mean']:.3f}")
    print(f"\nOutputs → {output_dir}/analysis/")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ToolReliBench real API runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--openai-key",  default=os.getenv("OPENAI_API_KEY",""))
    parser.add_argument("--gemini-key",  default=os.getenv("GEMINI_API_KEY",""))
    parser.add_argument("--grok-key",    default=os.getenv("GROK_API_KEY",""))
    parser.add_argument("--runs-per-task", type=int, default=3)
    parser.add_argument("--output-dir",  default="./real_results")
    parser.add_argument("--tasks",       nargs="*", default=None,
                        help="Subset of task IDs to run (e.g. short_000 long_001)")
    args = parser.parse_args()

    # Build client dict (only models where key is provided)
    clients = {}
    if args.openai_key:
        clients["gpt-4o-mini"] = OpenAIClient(args.openai_key)
        log.info("✓ OpenAI client ready (GPT-4o-mini)")
    if args.gemini_key:
        clients["gemini-1.5-flash"] = GeminiClient(args.gemini_key)
        log.info("✓ Gemini client ready (Gemini 1.5 Flash)")
    if args.grok_key:
        clients["grok-beta"] = GrokClient(args.grok_key)
        log.info("✓ Grok client ready (grok-beta via xAI)")

    if not clients:
        print("\n❌  No API keys provided. Provide at least one of:")
        print("   --openai-key  sk-...")
        print("   --gemini-key  AIza...")
        print("   --grok-key    xai-...")
        print("Or set env vars: OPENAI_API_KEY, GEMINI_API_KEY, GROK_API_KEY\n")
        sys.exit(1)

    # Filter tasks if specified
    global TASKS
    if args.tasks:
        TASKS = [t for t in TASKS if t["task_id"] in args.tasks]
        log.info(f"Running subset: {[t['task_id'] for t in TASKS]}")

    output_dir = Path(args.output_dir)
    all_traces, total_cost = run_experiment(clients, args.runs_per_task, output_dir)

    # Only analyse models that produced traces
    all_traces = {m: v for m, v in all_traces.items() if v}
    if all_traces:
        generate_report(all_traces, output_dir)
    else:
        log.error("No traces generated — check API keys and connectivity.")
        sys.exit(1)


if __name__ == "__main__":
    main()
