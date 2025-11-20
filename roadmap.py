from typing import List, Dict, Any


def build_roadmap(query: str, results: List[Dict[str, Any]], max_steps: int = 5) -> List[Dict[str, Any]]:
    # Heuristic grouping by existing label hierarchy if present, else by source, else buckets by similarity.
    # This is a simple baseline; can be replaced by a graph-based prerequisite model later.
    by_label: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        labels = item.get("labels") or []
        key = "/".join(labels[:2]) if labels else (item.get("source") or "general")
        by_label.setdefault(key, []).append(item)

    # Sort groups by best similarity in the group (descending)
    groups: list[tuple[str, list[dict[str, Any]]]] = sorted(
        by_label.items(),
        key=lambda kv: max(doc.get("similarity_score", 0.0) for doc in kv[1]),
        reverse=True,
    )

    roadmap_steps: list[dict[str, Any]] = []
    for i, (group_key, docs) in enumerate(groups[:max_steps], start=1):
        # Sort docs within step by credibility/score and then similarity
        docs_sorted = sorted(
            docs,
            key=lambda d: (float(d.get("credibility_score", 0.0)), float(d.get("similarity_score", 0.0))),
            reverse=True,
        )
        roadmap_steps.append(
            {
                "step": i,
                "title": group_key,
                "items": [
                    {
                        "title": d.get("title"),
                        "link": d.get("link"),
                        "source": d.get("source"),
                        "labels": d.get("labels", []),
                        "credibility_score": d.get("credibility_score"),
                        "similarity_score": d.get("similarity_score"),
                    }
                    for d in docs_sorted
                ],
            }
        )

    # If not enough groups, fill remaining steps by slicing the tail of results
    if len(roadmap_steps) < max_steps:
        remaining = results[: max(0, max_steps - len(roadmap_steps))]
        for j, item in enumerate(remaining, start=len(roadmap_steps) + 1):
            roadmap_steps.append(
                {
                    "step": j,
                    "title": item.get("title") or "Additional",
                    "items": [
                        {
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "source": item.get("source"),
                            "labels": item.get("labels", []),
                            "credibility_score": item.get("credibility_score"),
                            "similarity_score": item.get("similarity_score"),
                        }
                    ],
                }
            )

    return roadmap_steps


