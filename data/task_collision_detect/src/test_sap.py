class AABB:
    def __init__(self, obj_id, min_x, max_x):
        self.obj_id = obj_id
        self.min_x = min_x
        self.max_x = max_x

def sweep_and_prune_1d(objects):
    """
    1次元(X軸)のSweep and Pruneアルゴリズム
    :param objects: AABBオブジェクトのリスト
    :return: 衝突の可能性があるIDペアのセット
    """
    # 1. 最小値(min_x)が小さい順にオブジェクトをソート
    # (実際のゲームエンジン等では、フレーム間で位置が大きく変わらないため
    #  挿入ソート/Insertion Sort を使うことでO(N)に近い速度になります)
    sorted_objects = sorted(objects, key=lambda obj: obj.min_x)
    
    possible_pairs = set()
    active_list = []
    
    # 2. スキャン（掃き出し）処理
    for current in sorted_objects:
        # active_list の中で、currentのmin_xより前に完全に終わっているオブジェクトを除去（Prune）
        # ※ リストを逆引きして安全に削除
        for active in list(active_list):
            if active.max_x < current.min_x:
                active_list.remove(active)
        
        # 3. 現在active_listに残っているオブジェクトは、currentとX軸上で重なっている
        for active in active_list:
            # 重複を避けるため、IDの大小でペアを作る
            pair = (min(current.obj_id, active.obj_id), max(current.obj_id, active.obj_id))
            possible_pairs.add(pair)
            
        # 自身をアクティブリストに追加
        active_list.append(current)
        
    return possible_pairs

# --- テスト実行 ---
if __name__ == "__main__":
    # 3つのオブジェクトを定義 (ID, min_x, max_x)
    # A と B は重なっている
    # B と C は重なっている
    # A と C は重なっていない
    objA = AABB("A", 1.0, 5.0)
    objB = AABB("B", 4.0, 8.0)
    objC = AABB("C", 7.0, 12.0)
    
    scene_objects = [objC, objA, objB] # 順不同でリスト化
    
    # SAPの実行
    collision_pairs = sweep_and_prune_1d(scene_objects)
    
    print("衝突の可能性があるペア:")
    for pair in collision_pairs:
        print(f" - {pair[0]} と {pair[1]}")
