import os
import re
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# Không cần 'requests' hay 'time' nữa
# (THÊM MỚI) Thư viện cho Tính năng 4 (NLP)
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. KHỞI TẠO VÀ LÀM SẠCH ---
load_dotenv()
app = Flask(__name__)
CORS(app)


print("Đang tải và làm sạch dữ liệu...")
# (KHỐI NÀY ĐƯỢC SỬA LẠI)
try:
    df = pd.read_csv("data/anime_dataset.csv")

    def clean_string_list(text):
        if isinstance(text, str):
            text = re.sub(r"[\[\]'\"]", "", text)
            return text.strip()
        return text

    df['genres'] = df['genres'].apply(clean_string_list)
    df['studios'] = df['studios'].apply(clean_string_list)
    
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['members'] = pd.to_numeric(df['members'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    # (SỬA DÒNG NÀY) Thêm 'synopsis' vào
    df.dropna(subset=['score', 'members', 'year', 'genres', 'studios', 'episodes', 'synopsis'], inplace=True)
    
    df['genres'] = df['genres'].fillna('')
    df['studios'] = df['studios'].fillna('Unknown')
    df['synopsis'] = df['synopsis'].fillna('') # (THÊM MỚI)
    df['year'] = df['year'].astype(int)
    df['episodes'] = df['episodes'].astype(int)
    
    # === (SỬA LỖI TRÙNG LẶP) ===
    df.sort_values('members', ascending=False, inplace=True)
    df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    # (THÊM MỚI) Reset index để đảm bảo vị trí (iloc) khớp với index
    df = df.reset_index(drop=True) 
    print(f"Đã lọc trùng lặp, còn lại {len(df)} anime.")
    # === KẾT THÚC SỬA ===

    print("Tải dữ liệu thành công!")

except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG KHI TẢI DỮ LIỆU: {e}")
    exit()

# --- 2. (THÊM MỚI) TÍNH TOÁN TRƯỚC CHO TÍNH NĂNG 4 ---
print("Đang tính toán ma trận NLP (TF-IDF)...")
try:
    # 1. Tính toán NLP (Score_NLP)
    # Bỏ qua các từ tiếng Anh phổ biến, chỉ lấy từ xuất hiện trong ít nhất 5 tóm tắt
    # và không xuất hiện trong quá 80% tóm tắt (để bỏ từ quá chung chung)
    tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(df['synopsis'])
    
    # Đây là ma trận NxN chứa điểm tương đồng NLP của mọi anime với mọi anime khác
    cosine_sim_nlp = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # (THÊM MỚI) Lưu lại từ điển và chuyển thành mảng numpy
    tfidf_vocab = np.array(tfidf.get_feature_names_out())
    print("Ma trận NLP đã được tính toán xong!")

    # 2. Tạo Map để tra cứu (Title -> Vị trí index)
    # Rất quan trọng: giúp tìm (ví dụ) "Attack on Titan" là index số 5
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    # 3. Tính toán Phạm vi năm (Score_Year)
    max_year = df['year'].max()
    min_year = df['year'].min()
    # Tránh lỗi chia cho 0 nếu tất cả anime cùng 1 năm
    year_range = max_year - min_year if max_year - min_year > 0 else 1 

except Exception as e:
    print(f"LỖI KHI TÍNH TOÁN NLP: {e}")
    exit()

# --- 3. (THÊM MỚI) CÁC HÀM HỖ TRỢ TÍNH TOÁN F4 ---

# (THÊM MỚI) Hàm trích xuất từ khóa
def get_shared_keywords(vector1, vector2, vocab, top_n=3):
    """
    Tìm top_n từ khóa chung quan trọng nhất từ 2 vector TF-IDF
    """
    try:
        # Nhân 2 vector để tìm điểm chung
        combined_scores = vector1.multiply(vector2)
        
        # Lấy index của các từ có điểm (data)
        # Sắp xếp chúng và lấy top_n
        data_indices = combined_scores.data.argsort()[-top_n:][::-1]
        
        # Lấy index của các từ đó trong ma trận (features)
        feature_indices = combined_scores.indices[data_indices]
        
        # Lấy tên từ khóa từ vocab
        keywords = [vocab[i] for i in feature_indices]
        return keywords
    except Exception as e:
        print(f"Lỗi keyword: {e}")
        return []

def get_jaccard_sim(genres1_str, genres2_str):
    """Tính điểm Jaccard cho 2 chuỗi thể loại (0-1)"""
    try:
        set1 = set(g.strip() for g in genres1_str.split(',') if g.strip())
        set2 = set(g.strip() for g in genres2_str.split(',') if g.strip())
        
        if not set1 and not set2:
            return 1.0 # Nếu cả hai đều không có thể loại -> coi như giống nhau
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    except Exception:
        return 0.0 # Bỏ qua nếu có lỗi

def get_studio_sim(studios1_str, studios2_str):
    """Tính điểm Studio (1 nếu chung, 0 nếu không)"""
    try:
        if studios1_str == 'Unknown' or studios2_str == 'Unknown':
            return 0.0 # Không tính studio 'Unknown'
            
        set1 = set(s.strip() for s in studios1_str.split(',') if s.strip())
        set2 = set(s.strip() for s in studios2_str.split(',') if s.strip())
        
        return 1.0 if len(set1.intersection(set2)) > 0 else 0.0
    except Exception:
        return 0.0

def get_year_sim(year1, year2):
    """Tính điểm tương đồng năm (0-1)"""
    global year_range # Lấy biến toàn cục đã tính
    try:
        diff = abs(year1 - year2)
        return 1.0 - (diff / year_range)
    except Exception:
        return 0.0

# --- 4. API ENDPOINTS ---

@app.route('/')
def home():
# Gửi file index.html từ thư mục hiện tại (nơi app.py đang chạy)
 return send_from_directory('client', 'index.html')

# API cho TÍNH NĂNG 1: Top 5 Studios (ĐÃ SỬA)
@app.route('/api/top_studios', methods=['GET'])
def get_top_studios():
    genre_filter = request.args.get('genre')

    if genre_filter:
        # (SỬA DÒNG NÀY) Thêm \b và re.escape
        filtered_df = df[df['genres'].str.contains(r'\b' + re.escape(genre_filter) + r'\b', case=False, na=False)]
    else:
        filtered_df = df.copy()

    # 1. Logic Biểu đồ
    studio_stats = filtered_df[filtered_df['studios'] != 'Unknown'].groupby('studios').agg(
        avg_score=('score', 'mean'),
        anime_count=('title', 'count')
    )
    if not genre_filter:
        studio_stats = studio_stats[studio_stats['anime_count'] >= 3]
    top_studios = studio_stats.sort_values(by='avg_score', ascending=False).head(5).round(2)
    top_studios_data = top_studios.reset_index().to_dict(orient='records')
    top_studios_list = top_studios.index.tolist()


    # 2. Logic Grid Ảnh (Giờ đây siêu nhanh, không gọi Jikan)
    anime_grid_data = filtered_df[
        filtered_df['studios'].isin(top_studios_list)
    ].sort_values(by='score', ascending=False).head(10)
   
    anime_grid_dict = anime_grid_data.to_dict(orient='records')


    # Trả về JSON (không có 'image_url')
    return jsonify({
        "chart_data": top_studios_data,
        "grid_data": anime_grid_dict
    })


# API Phụ: Lấy danh sách thể loại
@app.route('/api/genres', methods=['GET'])
def get_genres():
    all_genres = set()
    for genre_list_str in df['genres'].str.split(','):
        for genre in genre_list_str:
            cleaned_genre = genre.strip()
            if cleaned_genre:
                all_genres.add(cleaned_genre)
   
    sorted_genres = sorted(list(all_genres))
    return jsonify(sorted_genres)

# === API MỚI CHO TÍNH NĂNG 2 (SỬA LỖI LỌC "AND") ===
@app.route('/api/filter', methods=['GET'])
def filter_anime():
    query = df.copy()

    # Lấy các bộ lọc (Lesson 7)
    genres_str = request.args.get('genres') # vd: "Action,Comedy"
    year = request.args.get('year', type=int)
    studio = request.args.get('studio')
    max_episodes = request.args.get('max_episodes', type=int)
    min_score = request.args.get('min_score', type=float)

    # === (SỬA LỖI LỌC "AND" - Tokyo Ghoul) ===
    # Logic "AND": Anime phải chứa TẤT CẢ các thể loại được chọn
    if genres_str:
        genre_list = [g.strip() for g in genres_str.split(',')]
        # Lặp qua từng thể loại và lọc (AND logic)
        for genre in genre_list:
            # dùng re.escape và \b (word boundary) để tìm chính xác
            query = query[query['genres'].str.contains(r'\b' + re.escape(genre) + r'\b', case=False, na=False)]
    # === KẾT THÚC SỬA LỖI ===
    
    if year:
        query = query[query['year'] == year]
    
    if studio:
     # (SỬA LỖI) Chỉ dùng re.escape. Bỏ \b để tìm được "AIC PLUS+"
        query = query[query['studios'].str.contains(re.escape(studio), case=False, na=False)]

    if max_episodes:
        query = query[query['episodes'] <= max_episodes]
        
    if min_score:
        query = query[query['score'] >= min_score]

    # Sắp xếp theo "hot" (members) và trả về 50 kết quả
    results = query.sort_values(by='members', ascending=False).head(50)
    return jsonify(results.to_dict(orient='records'))

# API Phụ: Lấy tất cả các bộ lọc cho F2
@app.route('/api/all_filters', methods=['GET'])
def get_all_filters():
    # Tái sử dụng hàm /api/genres
    genres = get_genres().json
    
    years = sorted(df['year'].unique().tolist(), reverse=True)
    studios = sorted(df[df['studios'] != 'Unknown']['studios'].unique().tolist())
    
    return jsonify({
        "genres": genres,
        "years": years,
        "studios": studios
    })

# === (THÊM MỚI) API CHO TÍNH NĂNG 4 ===

@app.route('/api/all_titles', methods=['GET'])
def get_all_titles():
    """API phụ: Lấy tất cả tên anime cho ô input"""
    titles = sorted(df['title'].tolist())
    return jsonify(titles)

# (THAY THẾ TOÀN BỘ HÀM NÀY)
@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    """API chính: Tính toán và trả về gợi ý (ĐÃ NÂNG CẤP)"""
    
    # 1. Lấy input (giữ nguyên)
    title_in = request.args.get('title')
    
    if not title_in:
        return jsonify({"error": "Thiếu tham số 'title'"}), 400
        
    if title_in not in indices:
        return jsonify({"error": f"Không tìm thấy anime '{title_in}' trong cơ sở dữ liệu."}), 404
        
    # 2. Lấy dữ liệu của anime gốc (giữ nguyên)
    idx_in = indices[title_in]
    anime_in = df.iloc[idx_in]
    genres_in = anime_in['genres']
    studios_in = anime_in['studios']
    year_in = anime_in['year']
    
    # (SỬA) Lấy set thể loại/studio của anime gốc
    set_genres_in = set(g.strip() for g in genres_in.split(',') if g.strip())
    set_studios_in = set(s.strip() for s in studios_in.split(',') if s.strip() and studios_in != 'Unknown')
    # (SỬA) Lấy vector NLP của anime gốc
    vector_in = tfidf_matrix[idx_in]

    # 3. Trọng số (giữ nguyên)
    W_NLP = 0.5
    W_GENRE = 0.3
    W_STUDIO = 0.1
    W_YEAR = 0.1
    
    scores_list = []
    
    # 4. Lặp qua TẤT CẢ anime (Lôgic bên trong đã được sửa)
    for idx_out in range(len(df)):
        if idx_out == idx_in:
            continue
            
        anime_out = df.iloc[idx_out]
        
        # --- Tính 4 điểm thành phần (đã sửa) ---
        
        # Score_NLP (tra cứu)
        score_nlp = cosine_sim_nlp[idx_in][idx_out]
        
        # Score_Genre (tính điểm + lấy thể loại chung)
        set_genres_out = set(g.strip() for g in anime_out['genres'].split(',') if g.strip())
        matching_genres_set = set_genres_in.intersection(set_genres_out)
        matching_genres_str = ", ".join(list(matching_genres_set))
        score_genre = get_jaccard_sim(genres_in, anime_out['genres']) # Vẫn dùng hàm cũ để tính điểm
        
        # Score_Studio (tính điểm + lấy studio chung)
        set_studios_out = set(s.strip() for s in anime_out['studios'].split(',') if s.strip() and anime_out['studios'] != 'Unknown')
        matching_studios_set = set_studios_in.intersection(set_studios_out)
        matching_studio_str = ", ".join(list(matching_studios_set))
        score_studio = 1.0 if len(matching_studios_set) > 0 else 0.0 # Tính điểm trực tiếp

        # Score_Year (giữ nguyên)
        score_year = get_year_sim(year_in, anime_out['year'])
        
        # --- (THÊM MỚI) Trích xuất lý do ---
        vector_out = tfidf_matrix[idx_out]
        shared_keywords = get_shared_keywords(vector_in, vector_out, tfidf_vocab, top_n=3)

        # --- Tính điểm cuối cùng (giữ nguyên) ---
        final_score = (score_nlp * W_NLP) + \
                      (score_genre * W_GENRE) + \
                      (score_studio * W_STUDIO) + \
                      (score_year * W_YEAR)
        
        # --- (SỬA) Gói tất cả dữ liệu vào dict ---
        result_item = anime_out.to_dict()
        result_item['similarity_score'] = round(final_score, 4)
        # Gói các điểm thành phần
        result_item['score_nlp'] = round(score_nlp, 4)
        result_item['score_genre'] = round(score_genre, 4)
        result_item['score_studio'] = round(score_studio, 4)
        result_item['score_year'] = round(score_year, 4)
        # Gói các lý do
        result_item['matching_keywords'] = [k.capitalize() for k in shared_keywords] # Viết hoa từ khóa
        result_item['matching_genres'] = matching_genres_str
        result_item['matching_studio'] = matching_studio_str

        scores_list.append((final_score, result_item))

    # 5. Sắp xếp và định dạng output (Sửa nhẹ)
    sorted_scores = sorted(scores_list, key=lambda x: x[0], reverse=True)
    
    # (SỬA) Dữ liệu cho Grid giờ đã chứa tất cả thông tin
    grid_data = [item_dict for score, item_dict in sorted_scores]
        
    # Dữ liệu cho Chart (giữ nguyên)
    top_5_raw = grid_data[:5]
    chart_data = [{
        "name": a['title'], 
        "y": a['similarity_score']
    } for a in top_5_raw]

    return jsonify({
        "source_anime": anime_in.to_dict(),
        "chart_data": chart_data,
        "grid_data": grid_data
    })
# === HẾT API TÍNH NĂNG 4 ===

# === API MỚI CHO TÍNH NĂNG 3 (Rubric 4) ===
@app.route('/api/genre_trends', methods=['GET'])
def get_genre_trends():
    # 1. Lấy danh sách thể loại từ query param (vd: ?genres=Action,Comedy,Drama)
    genres_str = request.args.get('genres')
    
    # Nếu không có thể loại nào được chọn, trả về rỗng
    if not genres_str:
        return jsonify({"categories": [], "series": []})
        
    genre_list = [g.strip() for g in genres_str.split(',')]
    
    # 2. Lọc dữ liệu theo năm (2010-2022)
    df_trend = df[(df['year'] >= 2010) & (df['year'] <= 2022)]
    years = sorted(df_trend['year'].unique().tolist())
    
    series_data = [] # Mảng để chứa các chuỗi (line)
    
    # 3. Lặp qua từng thể loại người dùng chọn và tính toán
    for genre in genre_list:
        # Tìm các anime chứa thể loại này
        genre_df = df_trend[df_trend['genres'].str.contains(r'\b' + re.escape(genre) + r'\b', case=False, na=False)]
        # Tính điểm trung bình theo năm
        scores = genre_df.groupby('year')['score'].mean().round(2)
        
        # Tạo dữ liệu chuỗi
        series_data.append({
            "name": genre,
            # Lấy điểm, nếu năm đó không có anime nào thì trả về 'None' (null)
            "data": [scores.get(year) for year in years] 
        })

    # 4. Trả về dữ liệu cho Highcharts
    return jsonify({
        "categories": years,
        "series": series_data
    })

# --- 5. CHẠY APP ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, port=port)