# Surrogate Modeling Project - Deployment Guide

## 🚀 Quick Start to Deploy on Hugging Face Spaces (FREE)

### Option 1: Deploy with One Click (Easiest)

1. **Go to**: [Hugging Face Spaces](https://huggingface.co/spaces)
2. **Click**: "Create new Space"
3. **Select**:
   - Space name: `surrogate-model-demo`
   - License: `mit`
   - **Dockerfile or Streamlit** → Choose **Streamlit**
4. **Copy this repository** into the Space
5. **Done!** 🎉 Your app will auto-deploy in ~2 minutes

**Result URL**: `https://huggingface.co/spaces/YOUR_USERNAME/surrogate-model-demo`

---

### Option 2: Manual GitHub Deploy

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Surrogate model deployment"
   git remote add origin https://github.com/YOUR-USERNAME/surrogate-model.git
   git push -u origin main
   ```

2. **Link to Hugging Face**:
   - Go to: [Create Space on HF](https://huggingface.co/spaces)
   - Select: **Streamlit** as SDK
   - Connect GitHub repo
   - Done! Auto-deploys on each push

---

### Option 3: Local Testing Before Deploy

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run app locally**:
   ```bash
   streamlit run app.py
   ```

3. **Visit**: http://localhost:8501

---

## 📁 Project Structure

```
Surrogate_Modeling_Project/
├── app.py                                  # Streamlit web interface
├── requirements.txt                        # Python dependencies
├── data/
│   ├── generate_synthetic_fem_data.py
│   └── fem_coupled_hygrothermal_mechanical.csv
├── models/
│   ├── train_surrogate_models.py
│   └── results/
│       ├── saved_models/
│       │   ├── gp_model.pkl              # GP trained model
│       │   ├── rf_model.pkl              # RF trained model
│       │   └── scaler.pkl                # Feature scaler
│       └── plots/
│           ├── 01_parity_plots.png
│           ├── 02_residuals_plots.png
│           ├── 03_feature_importance.png
│           └── 04_model_comparison.png
├── PROJECT_REPORT.md
├── TECHNICAL_NOTE_WITH_RESULTS.md
└── README.md (this file)
```

---

## 🌐 Web App Features

### Main Interface
- **Interactive sliders** for all 12 material/environmental parameters
- **Real-time predictions** from GP and RF models
- **Uncertainty quantification** (95% confidence intervals)
- **Safety assessment** with critical thresholds

### Output Sections
1. **Gaussian Process**: Central prediction + confidence bounds
2. **Random Forest**: Ensemble comparison
3. **Detailed Analysis**: 
   - Degradation breakdown
   - Feature sensitivity
   - Material characterization
4. **Export**: Download results as CSV/TXT

---

## 🎯 Deployment Comparison

| Platform | Cost | Setup Time | Pros | Cons |
|----------|------|-----------|------|------|
| **Hugging Face Spaces** | FREE | 2 min | Easiest, great UI, perfect for ML | Limited compute |
| **Streamlit Cloud** | FREE | 5 min | Native Streamlit support | May slow under load |
| **Railway.app** | FREE tier | 10 min | Full Python support, fast | Needs config file |
| **Heroku** | Paid ($7+/mo) | 5 min | Robust, widely used | No longer free tier |

**Recommendation**: **Hugging Face Spaces** ✅ (easiest, most popular for ML)

---

## 📊 App Walkthrough

### Step 1: Adjust Parameters (Sidebar)
- Use sliders to define your material and environmental conditions
- Values update predictions in real-time

### Step 2: View Predictions (Main Panel)
- **Left**: GP predictions with uncertainty
- **Right**: RF predictions + ensemble

### Step 3: Analyze Details (Tabs)
- **Degradation Analysis**: Moisture-mechanical effects
- **Feature Sensitivity**: Which parameters matter most
- **Material Info**: Full characterization

### Step 4: Export Results (Bottom)
- Download prediction summary as CSV or TXT
- Share with colleagues

---

## 🔧 Customization

### Change Model Paths
In `app.py`, line ~55:
```python
models_dir = Path(__file__).parent / 'results' / 'saved_models'
```

### Adjust Parameter Ranges
In `app.py`, update sliders:
```python
porosity = st.sidebar.slider("Porosity (%)", 0.2, 0.8, 0.5, 0.01)
```

### Modify Safety Thresholds
In `app.py`, line ~120:
```python
critical_threshold = 1.2  # Change this value
```

---

## 📈 Performance

- **Prediction latency**: < 50 ms per request
- **Concurrent users** (HF Spaces): ~10–20 simultaneously
- **Storage**: Models + data = ~50 MB
- **Inference speed**: 3,600–50,000× faster than FEM

---

## ⚠️ Troubleshooting

### Issue: "Models not found"
**Solution**: Ensure `results/saved_models/` directory exists with:
- `gp_model.pkl`
- `rf_model.pkl`
- `scaler.pkl`

**Action**: Run `python models/train_surrogate_models.py` locally first

### Issue: Slow on Hugging Face Spaces
**Solution**: Reduce model complexity or use cached predictions
```python
@st.cache_resource  # Already implemented!
def load_models():
    ...
```

### Issue: Import errors on deploy
**Solution**: Ensure all packages in `requirements.txt`
```bash
pip freeze > requirements.txt
```

---

## 🔗 Deployment Links

After deploying on HF Spaces, share this URL with Prof. Garbowski:

```
📍 Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/surrogate-model-demo
GitHub Repo: https://github.com/YOUR-USERNAME/surrogate-model
```

---

## 📚 Documentation Links

- **Project Report**: `PROJECT_REPORT.md`
- **Technical Note**: `TECHNICAL_NOTE_WITH_RESULTS.md`
- **Streamlit Docs**: https://docs.streamlit.io/
- **HF Spaces Guide**: https://huggingface.co/docs/hub/spaces

---

## 👥 For Prof. Tomasz Garbowski & Prof. Anna Szymczak-Graczyk

**Access the deployed model**:
1. Open the live URL (after deployment)
2. Adjust parameters for your material of interest
3. Get instant predictions with uncertainty
4. Export results for your analysis

**Suggested Use Cases**:
- ✅ Material screening (test multiple compositions)
- ✅ Sensitivity analysis (how does porosity affect stiffness?)
- ✅ Digital twin inputs (real-time monitoring)
- ✅ Student teaching tool (visualize coupled effects)

---

## 📧 Questions?

- **Deployment issues**: Check Hugging Face Spaces documentation
- **Model questions**: See `PROJECT_REPORT.md` or `TECHNICAL_NOTE_WITH_RESULTS.md`
- **Customization**: Available on request

---

**Project Status**: ✅ Ready for Deployment

Last Updated: March 14, 2026
