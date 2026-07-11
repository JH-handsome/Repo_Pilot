import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_repository_package_exports_codebase_management(self):
        from coding_rag.repository.chunks import CodeChunk
        from coding_rag.repository.files import PythonFile
        from coding_rag.repository.index import RepoIndex
        from coding_rag.repository import CodeChunk, PythonFile, RepoIndex

        self.assertIsNotNone(CodeChunk)
        self.assertIsNotNone(PythonFile)
        self.assertIsNotNone(RepoIndex)

    def test_tools_package_exports_retrieval_tools(self):
        from coding_rag.tools.bm25 import BM25Retriever
        from coding_rag.tools.filter import filter_recalled_results
        from coding_rag.tools.tokenizer import CodeTokenizer
        from coding_rag.tools import BM25Retriever, CodeTokenizer, filter_recalled_results

        self.assertIsNotNone(BM25Retriever)
        self.assertIsNotNone(CodeTokenizer)
        self.assertIsNotNone(filter_recalled_results)

    def test_rag_package_exports_ask_pipeline(self):
        from coding_rag.rag.ask import AskModeConfig
        from coding_rag.rag import AskModeConfig, GenerationMode
        from coding_rag.rag.prompt import GenerationMode

        self.assertIsNotNone(AskModeConfig)
        self.assertIsNotNone(GenerationMode)

    def test_agent_package_exports_workflow_and_memory(self):
        from coding_rag.agent import CodeAgentConfig, AgentMemory
        from coding_rag.agent.memory import AgentMemory
        from coding_rag.agent.planner import WorkflowMode
        from coding_rag.agent.workflow import CodeAgentConfig

        self.assertIsNotNone(CodeAgentConfig)
        self.assertIsNotNone(AgentMemory)
        self.assertIsNotNone(WorkflowMode)

    def test_rag_package_owns_llm_client(self):
        from coding_rag.rag.llm_client import LLMConfig

        self.assertIsNotNone(LLMConfig)


if __name__ == "__main__":
    unittest.main()
