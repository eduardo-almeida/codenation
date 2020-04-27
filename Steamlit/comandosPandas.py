import streamlit as st
import pandas as pd

def main():
    st.title('ola teste')
    file = st.file_uploader('Escolha seu arquivo', type='csv')
    if file is not None:
        slider = st.slider('Valores:', 0, 100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('Separador')
        st.table(df.head(slider))
        st.write(df.columns)
        st.markdown('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        st.table(df.groupby('class')['petal width'].mean())

if __name__ == '__main__':
    main()