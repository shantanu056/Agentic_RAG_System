from app.orchestration.rag_pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline()

    while True:

        query = input("\n Enter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        response = pipeline.run(query)

        print(f"\n Query Type: {response['query_type']}")

        if response["strategy"]:
            print(
                f"\n Retrieval Strategy: "
                f"{response['strategy']}"
            )

        print("\n Answer:\n")
        print(response["answer"])

        if response["validation"]:
            print("\n Validation Report:\n")
            print(response["validation"])


if __name__ == "__main__":
    main()