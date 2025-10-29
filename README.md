IntelliTradeX
  AI-Powered Stock Prediction & Sentiment Analysis Dashboard
Problem Statement Id : SIH25127

🧠 Abstract:
In today’s fast-paced financial markets, predicting stock movements and understanding investor sentiment are crucial for informed decision-making. IntelliTradeX is an AI-driven web application that integrates Machine Learning models for stock price prediction with Natural Language Processing (NLP) for sentiment analysis of financial news and market data.
The system enables users to analyze, predict, and visualize real-time stock trends while providing AI-based insights on the emotional tone of related news headlines. The integration of backend intelligence with a dynamic, interactive frontend allows users to explore market behavior intuitively and make data-driven investment decisions.

🧩 System Overview:
🔹 Frontend 
Built using React.js (or Next.js with Tailwind CSS and ShadCN UI)
Provides an interactive dark-themed dashboard
Handles user interactions, data visualization, and real-time updates
Connects to the backend via Axios API calls
Uses Recharts and Framer Motion for elegant animations and charting

🔹 Backend 
Built with FastAPI (Python) for high-performance API handling
Contains ML models for:
Stock Price Prediction (using regression or LSTM models)
Sentiment Analysis (using NLP and pre-trained models)
Stores and retrieves prediction records, training logs, and user queries
Interfaces with a database (MongoDB / PostgreSQL / SQLite) for persistent storage
Includes modular Python services:
model_service.py → Handles model loading, training, and prediction
sentiment_service.py → Performs sentiment analysis on news text
data_fetcher.py → Fetches live stock data via APIs (e.g., Yahoo Finance, Alpha Vantage)
db.py → Manages database sessions and schema models

TECH STACK
Frontend: Built with React.js / Next.js, styled using Tailwind CSS and ShadCN UI. Tools like Recharts, Axios, and Framer Motion handle data visualization, API communication, and animations, ensuring a smooth and dynamic user experience.
Backend: Developed in Python using FastAPI for high-performance API handling. It integrates Pydantic for validation, Loguru for logging, Uvicorn as the server, and Dotenv for secure environment management.
Machine Learning: Uses Scikit-learn, TensorFlow, Pandas, NumPy, and Joblib to power the hybrid LSTM–Random Forest model for stock prediction and sentiment analysis.
Database: SQLite / MongoDB stores prediction results, sentiment logs, and training data for continuous improvement.
Development & Version Control: Tools like VS Code, Postman, and Git were used for development, while GitHub manages code versioning and collaboration

🧠 System Architecture

The architecture of IntelliTradeX follows a modular and data-driven pipeline integrating multiple components:

Frontend (Streamlit)

Acts as the user-facing dashboard.

Displays visual insights like price trends, sentiment scores, and prediction graphs.

Backend (Flask + Python)

Processes requests from the frontend.

Connects with the ML models, APIs, and database for responses.

AI/ML Layer

Built using TensorFlow, Keras, and Scikit-learn.

Implements LSTM (Long Short-Term Memory) networks for stock price forecasting.

Integrates Sentiment Analysis (VADER / OpenAI API) to gauge market mood.

Data Sources

Uses Yahoo Finance API for historical market data.

Uses News API and VADER for sentiment and trend extraction.

Database (SQLite)

Stores stock history, trained model logs, and prediction records.

Visualization & Insights

Displays combined financial and emotional indicators through Matplotlib and Plotly charts.

💡 Innovative Features:

Dual AI systems (Stock ML + Sentiment NLP)
Real-time prediction and visualization
Retrainable AI model
Smooth and interactive dark-mode UI
Data-driven decision insights for users
Modular and scalable backend architecture

📊  Data Flow:

1. User enters stock name in frontend → /predict
2. FastAPI receives input → queries ML model → returns predicted price
3. Frontend visualizes prediction on a dynamic Recharts line graph
4. User checks sentiment of related news → /sentiment
5. Backend NLP model analyzes tone → returns sentiment label
6. Dashboard updates live with insights
🔮 Future Enhancements:
•	Integrate Deep Learning (LSTM) for time-series forecasting
•	Add user authentication and saved portfolios
•	Enable real-time socket streaming for live prices
•	Deploy on AWS or Render for global accessibility

🏁 Conclusion:
IntelliTradeX combines the power of Machine Learning, Natural Language Processing, and Interactive Web Technologies to provide a unified AI system for financial analysis. The seamless integration of backend intelligence with a modern, futuristic frontend creates a professional-grade platform for market research, financial prediction, and sentiment-based decision support.
 

TEAM LEAD:  P. Krishna Keerthana
TEAM MEMBERS:  R. Laxmi Rupali
               R. Lalithasahasra
               V. Manoj
               G. Malishka Reddy
               R. Pavan Kumar
