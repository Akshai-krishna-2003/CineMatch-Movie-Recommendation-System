
# 🎬 CineMatch: Personalized Movie Recommendation System

**CineMatch** is a hybrid movie recommendation system that uses both content-based and collaborative filtering techniques to generate highly personalized movie recommendations. Built using the popular **MovieLens 20M dataset**, the system understands user preferences, overcomes the cold-start problem, and scales to large data efficiently.

---

## 📽️ Project Demo

<video src="https://github.com/user-attachments/assets/2ee48cf6-569c-43a9-9c42-0461506d0513" controls width="100%" style="border-radius: 10px;">
  Your browser does not support the video tag.
</video>

---

## 🔍 How It Works

CineMatch uses a **hybrid recommendation engine**:

1. **Content-Based Filtering (TF-IDF + Cosine Similarity)**  
   - Uses genres, tags, and descriptions of movies  
   - Applies **TF-IDF vectorization** to convert text into vectors  
   - Calculates similarity using **cosine similarity**

2. **Collaborative Filtering (SVD)**  
   - Builds a **user-item matrix** from ratings  
   - Applies **Singular Value Decomposition (SVD)**  
   - Predicts ratings based on similar users’ behavior

3. **Hybrid Strategy**  
   - Combines predictions from both models  
   - Ranks and recommends the top N movies for each user

---

## ⚙️ Features

- 🔄 Hybrid recommendation combining **TF-IDF + SVD**
- 🧠 Personalized recommendations for every user
- ❄️ Handles **cold-start** and **sparse data** challenges
- 📊 Evaluation using **Precision**, **Recall**, and **F1-score**
- 🌐 Interactive **Flask web interface**
- ⚡ Scalable and lightweight

---

## 📊 Dataset

**Dataset**: [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)  
**Provider**: GroupLens Research, University of Minnesota

- 27,000+ movies  
- 138,000+ users  
- 20 million+ ratings  
- Includes metadata like genres, tags, and timestamps

---

## 📈 Evaluation Metrics

| Metric       | Description |
|--------------|-------------|
| **Precision** | Percentage of recommended movies that are actually relevant |
| **Recall**    | Percentage of relevant movies that are recommended |
| **F1-Score**  | Harmonic mean of precision and recall |

---

## 🚀 Getting Started

Set up CineMatch on your machine in a few minutes!

### ✅ 1. Prerequisites

- Python 3.9 – 3.11 (Recommended: Python 3.10 or above)
- `pip` (Python package manager)

### 📁 2. Clone and Navigate to the Project

```bash
git clone https://github.com/yourusername/CineMatch-Movie-Recommendation-System.git
cd CineMatch-Movie-Recommendation-System/content-recommender
```

### 📦 3. Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install flask
pip install numpy
pip install pandas
pip install scikit-learn==1.2.2
pip install joblib
```

✅ Use `scikit-learn==1.2.2` to match the version used for saving `TfidfVectorizer` and `SVD` models.

### 🧠 4. (Optional) Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux
```

Then install dependencies again inside the virtual environment.

### ▶️ 5. Run the Flask App

```bash
python app.py
```

Now open your browser and go to:

```
http://127.0.0.1:5000
```

You’ll see the CineMatch web interface up and running 🎉

---

## ⚠️ Common Warning

If you see this:

```
InconsistentVersionWarning: Trying to unpickle estimator TfidfVectorizer from version 1.2.2 when using version 1.6.1...
```

It’s because the pre-trained models were saved with **scikit-learn 1.2.2**, but you’re using a newer version.

### 🔧 Fix:

- Either downgrade:
  ```bash
  pip install scikit-learn==1.2.2
  ```
- Or retrain and re-save the models using your current version.

---

## 🧰 Tech Stack

- **Python 3.x**
- **Flask** – Web framework
- **scikit-learn** – ML models (TF-IDF, SVD)
- **Pandas**, **NumPy** – Data handling
- **Joblib** – Model saving/loading
- **HTML + CSS** – Frontend UI

---

## 🧑‍🎓 Author Details

- **Project Title**: CineMatch – Personalized Movie Recommendation System  
- **Name**: Akshai Krishna A  

---

## 🚧 Future Work

- 🌐 Add a complete frontend with user login
- 🔁 Implement user feedback loop
- 🧠 Upgrade to neural collaborative filtering
- 🌍 Add region/language-specific recommendation logic
- 📱 Deploy as a full-stack web or mobile app

---

## 🙌 Acknowledgements

- Thanks to my **supervisor** [Supervisor's Name] for continuous support  
- Appreciation to **friends, faculty, and peers** for suggestions  
- Credits to **GroupLens Research** for providing the dataset

---

> 🎥 “CineMatch – Watch what you'll love next!”




