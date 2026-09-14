import numpy as np

def support(shape1, shape2, d):
    """
    方向ベクトル d における2つの形状のミンコフスキー差の最遠点を返す
    """
    # shape1 から d 方向の最遠点を見つける
    p1 = shape1[np.argmax(np.dot(shape1, d))]
    # shape2 から -d 方向の最遠点を見つける
    p2 = shape2[np.argmax(np.dot(shape2, -d))]
    # ミンコフスキー差の点を返す
    return p1 - p2

def handle_simplex(simplex, d):
    """
    現在の単体（Simplex）の状態から、原点が含まれているかを判定し、
    次の探索方向 d を更新する（2D空間用）
    """
    if len(simplex) == 2:  # 線分の場合
        B, A = simplex[0], simplex[1]
        AB = B - A
        AO = -A
        
        # ABの法線ベクトル（外側向き）
        ABperp = np.array([-AB[1], AB[0]])
        if np.dot(ABperp, AO) < 0:
            ABperp = -ABperp
            
        d[:] = ABperp
        return False
        
    elif len(simplex) == 3:  # 三角形の場合
        C, B, A = simplex[0], simplex[1], simplex[2]
        AB = B - A
        AC = C - A
        AO = -A
        
        # ABの法線
        ABperp = np.array([-AB[1], AB[0]])
        if np.dot(ABperp, AC) > 0:
            ABperp = -ABperp
            
        # ACの法線
        ACperp = np.array([-AC[1], AC[0]])
        if np.dot(ACperp, AB) > 0:
            ACperp = -ACperp
            
        # 原点がABの外側にある場合
        if np.dot(ABperp, AO) > 0:
            simplex.pop(0)  # Cを削除して線分にする
            d[:] = ABperp
            return False
        # 原点がACの外側にある場合
        elif np.dot(ACperp, AO) > 0:
            simplex.pop(1)  # Bを削除して線分にする
            d[:] = ACperp
            return False
        else:
            # 三角形の中に原点がある = 衝突している
            return True
            
    return False

def gjk(shape1, shape2):
    """
    GJKアルゴリズムのメインループ
    衝突していれば True、していなければ False を返す
    """
    # 最初の探索方向（適当にX軸正の方向）
    d = np.array([1.0, 0.0])
    
    # 最初のサポート点を取得し、単体を初期化
    simplex = [support(shape1, shape2, d)]
    
    # 次の探索方向は、最初の点から原点へ向かう方向
    d = -simplex[0]
    
    while True:
        # 新しいサポート点を取得
        A = support(shape1, shape2, d)
        
        # 新しい点が原点を超えていなければ、衝突する可能性はない
        if np.dot(A, d) < 0:
            return False
            
        simplex.append(A)
        
        # 単体の更新と衝突判定
        if handle_simplex(simplex, d):
            return True

# --- 動作確認用のテストデータ ---

# 形状A（四角形）
polygon_A = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 2.0],
    [0.0, 2.0]
])

# 形状B（衝突する四角形）
polygon_B = np.array([
    [1.5, 1.5],
    [3.5, 1.5],
    [3.5, 3.5],
    [1.5, 3.5]
])

# 形状C（離れている四角形）
polygon_C = np.array([
    [4.0, 4.0],
    [6.0, 4.0],
    [6.0, 6.0],
    [4.0, 6.0]
])

print("A と B の衝突:", gjk(polygon_A, polygon_B)) # 期待値: True
print("A と C の衝突:", gjk(polygon_A, polygon_C)) # 期待値: False
