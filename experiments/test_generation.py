from src.generation.llm import generate_answer


question = input("Enter your question: ")

answer = generate_answer(question)

print("\nQuestion:")
print(question)

print("\nLLM Answer:")
print(answer)