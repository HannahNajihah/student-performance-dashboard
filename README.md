# 📊 Student Performance Dashboard  

This project presents a **full-stack data science application** that combines **Streamlit** for interactive visualisation and **Power BI** for in-depth reporting.  

The dashboard analyses **student academic performance** with respect to socio-demographic factors, providing **actionable insights** for educators and policymakers.  

---

## ✨ Features  

- **Interactive Dashboard**  
  Built with Streamlit and integrated with Power BI for dynamic and user-friendly exploration of student performance data.  

- **Automated Data Processing**  
  Preprocessing pipeline developed with modular Python scripts for cleaning, transformation, and outlier detection.  
  Continuous Integration and Deployment (CI/CD) using GitHub Actions ensures reproducibility and automated updates.  

- **Agile Development**  
  Managed with Asana and GitHub Projects to track progress and ensure iterative delivery.  
  Collaborative workflow following Agile best practices for team coordination.  

---

## 🖥️ Running the Dashboards  

There are two versions of the Streamlit dashboard in the `dashboards/` folder:  

- `mvp_dashboard.py` → Minimal version with data filtering and statistics only  
- `full_dashboard.py` → Full version with prediction using a trained model  

### To run the MVP dashboard:  

```bash
streamlit run dashboards/mvp_dashboard.py

### To run the full dashboard:  

streamlit run dashboards/full_dashboard.py

