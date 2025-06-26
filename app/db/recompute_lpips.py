import os
from db.database import Database
from modules.metrics import get_or_compute_lpips

def recompute_lpips_for_chat(chat_id):
    with Database() as db:
        images = db.fetch_images_by_chat(chat_id, pandas=True, include_depth=True).sort_values("depth")
        # LPIPS is only meaningful between selected images
        images = images[images["selected"] == 1]
        
        max_depth = int(images["depth"].max())
        for depth in range(1, max_depth + 1):
            current = images[images["depth"] == depth]
            previous = images[images["depth"] == depth - 1]

            if current.empty or previous.empty:
                continue

            curr = current.iloc[0]
            prev = previous.iloc[0]

            # print("user_id:", curr["user_id"], type(curr["user_id"]))

            if not os.path.exists(curr["path"]) or not os.path.exists(prev["path"]):
                print(f"Missing image file: {curr['path']} or {prev['path']}")
                print(f"Skipped LPIPS for depth {depth}: missing image file\n"
                      f"   → curr: {curr['path']}\n"
                      f"   → prev: {prev['path']}")

                continue

            # Check if LPIPS already exists in DB
            existing = db.fetch_lpips_by_chat(chat_id)
            match = existing[
                (existing["image_id"] == curr["id"]) &
                (existing["previous_image_id"] == prev["id"])
            ]
            if not match.empty:
                continue

            lpips_val = get_or_compute_lpips(curr["id"], curr["path"], prev["path"])
            db.insert_lpips_metric(curr["id"], 
                                   prev["id"], 
                                   int(curr["user_id"]),
                                   int(chat_id),
                                   int(depth),
                                   float(lpips_val))
