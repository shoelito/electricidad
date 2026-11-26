"""Punto de entrada del frontend. Ejecutar desde frontend/:  streamlit run app.py"""

import streamlit as st

st.set_page_config(page_title="Monitor eléctrico", page_icon="⚡", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/dashboard.py", title="Dashboard", icon="📈", default=True),
        st.Page("views/api_explorer.py", title="Explorador de la API", icon="🧪"),
    ]
)
navigation.run()
