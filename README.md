# Coding RAG Minimal Skeleton

A tiny Coding RAG retrieval project for learning.

Included modules:

- `file_loader`: recursively reads Python files
- `code_splitter`: splits code by line ranges
- `tokenizer`: code-aware tokenizer for BM25
- `bm25_retriever`: retrieves code chunks with `rank-bm25`
- `llm_client`: tiny OpenAI-compatible chat-completions client
- `llm_judge`: optional LLM judgment over BM25 results
- `rag/prompt`: prompt templates for various generation modes
- `rag/answer_generator`: answer generation with multiple modes
- `main.py`: command-line entrypoint

Not included: agents, vector databases, or WebUI.

## Install

```bash
pip install -r requirements.txt
```

## BM25 Search And Recall

```bash
python main.py /path/to/your/repo "your query" --top-k 5
```

Example:

```bash
python main.py datasets "two sum hash map" --top-k 3
```

The default retrieval flow is:

1. BM25 gets `--top-k` seed chunks.
2. Recall expands each seed with nearby chunks from the same file.
3. Final filtering re-scores recalled chunks and keeps the best `--final-k`.
4. The CLI prints the final context, and the optional LLM judges that context.

Control the second recall stage:

```bash
python main.py datasets "two sum hash map" --top-k 5 --recall-window 1
python main.py datasets "two sum hash map" --top-k 5 --recall-window 0
```

For a wider first pass before recall:

```bash
python main.py datasets "linked list cycle" --candidate-k 20 --recall-window 1 --max-recall-results 30
```

Control the final post-recall filter:

```bash
python main.py datasets "binary tree TreeNode" --candidate-k 20 --recall-window 1 --final-k 5
python main.py datasets "binary tree TreeNode" --candidate-k 20 --recall-window 1 --no-final-filter
```

Debug tokenizer output:

```bash
python main.py datasets "binary tree TreeNode level order" --top-k 5 --show-tokens
```

For code search, the tokenizer keeps full identifiers and also splits common
code names:

- `build_binary_tree` -> `build_binary_tree`, `build`, `binary`, `tree`
- `twoSum` -> `twosum`, `two`, `sum`
- Chinese queries are expanded into short n-grams

## LLM-Powered Answer Generation

BM25 returns candidate code chunks. The optional LLM step can generate answers
in multiple modes based on your needs.

### Generation Modes

| Mode | Description | Use Case |
| --- | --- | --- |
| `judge` | Judge relevance and answer questions | Search and understand code |
| `code-understand` | Explain code in depth | Learn how code works |
| `code-generate` | Generate code based on examples | Create new implementations |
| `leetcode` | Solve LeetCode problems | Algorithm problem solving |
| `api` | Generate API usage examples | Learn how to use APIs |

### Usage Examples

**Search and judge (default mode):**

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "two sum hash map" --top-k 5 --llm --llm-provider deepseek
```

**Code understanding:**

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "binary tree level order traversal" --top-k 5 --llm --mode code-understand
```

**Code generation:**

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "implement depth first search" --top-k 5 --llm --mode code-generate
```

**LeetCode problem solving:**

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "two sum" --top-k 5 --llm --mode leetcode
```

**API usage examples:**

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "how to use BM25 retriever" --top-k 5 --llm --mode api
```

### Legacy Mode

To use the old LLM judge behavior for backward compatibility:

```powershell
python main.py datasets "your query" --top-k 5 --llm --legacy-judge
```

### LLM Configuration

The client uses OpenAI-compatible `/chat/completions` HTTP APIs and does not add
extra dependencies.

Domestic provider presets:

| Provider | Default model | API key env |
| --- | --- | --- |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `qwen` | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `kimi` | `kimi-k2.6` | `MOONSHOT_API_KEY` |
| `zhipu` | `glm-4.7` | `ZHIPU_API_KEY` |

PowerShell examples:

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py datasets "two sum hash map" --top-k 5 --llm --llm-provider deepseek
```

```powershell
$env:DASHSCOPE_API_KEY="your_key"
python main.py datasets "binary tree level order traversal" --top-k 5 --llm --llm-provider qwen
```

Custom OpenAI-compatible endpoint:

```powershell
$env:LLM_API_KEY="your_key"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="your-model"
python main.py datasets "linked list cycle" --llm --llm-provider custom
```

You can override preset values:

```powershell
python main.py datasets "trie prefix search" --llm --llm-provider kimi --llm-model moonshot-v1-8k
```

**Control output length:**

```powershell
python main.py datasets "binary tree" --llm --llm-max-tokens 4000
```

**Adjust context size:**

```powershell
python main.py datasets "complex algorithm" --llm --llm-context-chars 20000
```

### Available LLM Providers

LeetCode defines `ListNode`, `TreeNode`, and several different `Node` shapes in
its judge environment. Local copied solutions may fail unless those names are
provided.

This project includes `leetcode_types.py` as a small local shim. After
downloading a LeetCode solution repo, patch it with:

```bash
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

## Desktop Frontend Window

If you already prepared prompts and want a visual window, run:

```bash
python web_ui.py
```

Then you can:

- choose repo directory
- input your prompt/query
- adjust retrieval parameters (`top-k`, `chunk-size`, `overlap`, `recall-window`)
- optionally enable LLM generation and pick mode/provider

The output panel shows recalled code chunks and optional LLM response.

## Tests

```bash
python -m unittest
python -m compileall coding_rag main.py tests
```
