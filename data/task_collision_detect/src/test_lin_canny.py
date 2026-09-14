import numpy as np

class LinCanny2DMinimal:
    """
    Lin-Cannyの概念を理解するための2D簡易モデル。
    点（別オブジェクトの頂点）と、線分（対象オブジェクトの辺）の最短特徴を追跡する。
    """
    @staticmethod
    def check_voronoi_region(point, edge_start, edge_end):
        """
        点(point)が、線分(edge)の『面の領域』にあるか、それとも『頂点の領域』に外れているかを判定。
        """
        ab = edge_end - edge_start
        ap = point - edge_start
        bp = point - edge_end
        
        # 1. 始点側の頂点領域に外れているか（内積が負）
        if np.dot(ap, ab) < 0:
            return "Vertex_Start"  # 始点側の頂点が新しい「最も近い特徴」候補
            
        # 2. 終点側の頂点領域に外れているか（内積が正）
        if np.dot(bp, ab) > 0:
            return "Vertex_End"    # 終点側の頂点が新しい「最も近い特徴」候補
            
        # 3. どちらの頂点からも外れていなければ、この辺（面）の領域内にある
        return "Edge_Face"

# --- 動作確認 ---
# 辺（多面体の一つの面に見立てる）
edge_s = np.array([0.0, 0.0])
edge_e = np.array([10.0, 0.0])

# テストする点（もうひとつのオブジェクトの頂点）
p_on_face = np.array([5.0, 3.0])   # 辺の真上（面のボロノイ領域内）
p_out_start = np.array([-2.0, 1.0]) # 始点より外側（始点頂点のボロノイ領域内）

print("点1の判定:", LinCanny2DMinimal.check_voronoi_region(p_on_face, edge_s, edge_e))
# 出力: Edge_Face（現在の面ペアを維持、またはここから距離を計算して衝突判定）

print("点2の判定:", LinCanny2DMinimal.check_voronoi_region(p_out_start, edge_s, edge_e))
# 出力: Vertex_Start（最も近い特徴が「辺」から「始点頂点」へ遷移したため、次のループでそちらをチェック）
