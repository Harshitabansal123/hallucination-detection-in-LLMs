
import gzip
import json

from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi

from .tokenizer import tokenize


class MIRACLBengaliShardRetriever:

    def __init__(self):
        self.language = "bn"
        self.num_shards = 1
        self.shards = []

        self.build_indexes()

    def load_shard(self, shard_number):
        filename = (
            f"miracl-corpus-v1.0-{self.language}/"
            f"docs-{shard_number}.jsonl.gz"
        )

        print(
            f"Loading MIRACL Bengali shard "
            f"{shard_number + 1}/{self.num_shards}..."
        )

        file_path = hf_hub_download(
            repo_id="miracl/miracl-corpus",
            filename=filename,
            repo_type="dataset"
        )

        passages = []

        with gzip.open(
            file_path,
            "rt",
            encoding="utf-8"
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                item = json.loads(line)

                passages.append({
                    "docid": item["docid"],
                    "title": item["title"],
                    "text": item["text"]
                })

        return passages

    def build_indexes(self):
        for shard_number in range(self.num_shards):
            passages = self.load_shard(shard_number)

            print(
                f"Building BM25 index for Bengali shard "
                f"{shard_number + 1} "
                f"({len(passages)} passages)..."
            )

            documents = [
                tokenize(
                    item["title"] + " " + item["text"]
                )
                for item in passages
            ]

            bm25 = BM25Okapi(documents)

            self.shards.append({
                "passages": passages,
                "bm25": bm25
            })

            print(
                f"Bengali shard {shard_number + 1} ready."
            )

        print(
            f"\nMIRACL Bengali retrieval index ready "
            f"with {len(self.shards)} shard(s)."
        )

    def retrieve(self, claim, top_k=200):
        if not isinstance(claim, str) or not claim.strip():
            return []

        if top_k <= 0:
            return []

        query_tokens = tokenize(claim)

        if not query_tokens:
            return []

        best_results = []

        for shard in self.shards:
            passages = shard["passages"]
            bm25 = shard["bm25"]

            scores = bm25.get_scores(query_tokens)

            candidate_count = min(top_k, len(scores))

            if candidate_count == 0:
                continue

            top_indices = scores.argsort()[
                -candidate_count:
            ][::-1]

            for index in top_indices:
                score = float(scores[index])

                if score <= 0:
                    continue

                item = passages[index].copy()
                item["relevance_score"] = score
                item["language"] = "bn"

                best_results.append(item)

        best_results.sort(
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        best_results = best_results[:top_k]

        for rank, item in enumerate(best_results, start=1):
            item["rank"] = rank

        return best_results

    def retrieve_many(self, claims, top_k=200):
        return [
            self.retrieve(claim, top_k=top_k)
            for claim in claims
        ]



if __name__ == "__main__":
    retriever = MIRACLBengaliShardRetriever()

    test_claims = [
        "বাংলাদেশের রাজধানী ঢাকা।",
        "রবীন্দ্রনাথ ঠাকুর নোবেল পুরস্কার পেয়েছিলেন।",
        "পৃথিবী সূর্যের চারদিকে ঘোরে।",
        "বাংলাদেশের জাতীয় ফুল শাপলা।",
        "ঢাকা বাংলাদেশের রাজধানী নয়।"
    ]

    for claim in test_claims:
        print("\n" + "=" * 70)
        print("CLAIM:", claim)

        results = retriever.retrieve(claim, top_k=5)

        if not results:
            print("No evidence retrieved.")
            continue

        for result in results:
            print(f"\nRank: {result['rank']}")
            print(f"Title: {result['title']}")
            print(f"Score: {result['relevance_score']:.4f}")
            print(f"Text: {result['text'][:250]}")
