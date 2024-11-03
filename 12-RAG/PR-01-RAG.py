
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

# LCEL 문서 로드 (내용이 많은 문서)
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text, encoding='utf-8'
)
docs = loader.load()
print(f"문서의 수: {len(docs)}")

# Loader에서 자동으로 넣어준 metadata를 삭제
docs[0].metadata = {}

# 문서 청킹
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print("문서 청크 개수: ", len(splits))

for i, split in enumerate(splits):
    # 검색된 chunk 구별을 위해 metadata에 id값을 삽입
    split.metadata["id"] = i

# 벡터스토어를 생성합니다.
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(), 
    collection_metadata={"hnsw:space": "cosine"}    # cosine 유사도 알고리즘을 사용합니다.
    )

# 문서를 저장합니다.
vector_ids = vectorstore.add_documents(splits)

# 예시 질의와 관련된 문서를 검색합니다.
query_example = "LangChain Expression Language (LCEL)는 무엇을 위해 설계되었나요?"
references_k_4 = vectorstore.similarity_search_with_score(query=query_example, k=4)
references_k_10 = vectorstore.similarity_search_with_score(query=query_example, k=10)

# 유사도 점수 내림차순으로 정렬
references_k_4.sort(key=lambda x:x[1], reverse=True)
references_k_10.sort(key=lambda x:x[1], reverse=True)

# 검색 결과 출력
print(" *** k=4 결과:\n")
for ref in references_k_4:
    print(ref[0])
    print("점수: ", ref[1])
    print("---------------------------------------")

print(" *** k=10 결과:\n")
for ref in references_k_10:
    print(ref[0])
    print("점수: ", ref[1])
    print("---------------------------------------")