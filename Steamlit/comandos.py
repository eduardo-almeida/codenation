import streamlit as st

def main():
    st.title('ola teste')
    st.markdown('Botao')
    botao = st.button('Botao')
    if botao:
        st.markdown('clicado')
    checkbox = st.checkbox('checkbox')
    if checkbox:
        st.markdown('clicado no checkbox')
    radio = st.radio('Escolha as opções', ('option1', 'option2'))
    if radio == 'option1':
        st.markdown('teste1')
    if radio == 'option2':
        st.markdown('teste2')
    st.markdown('selectbox')
    selectbox = st.selectbox('Escolha as opções', ('option1', 'option2'))
    if selectbox == 'option1':
        st.markdown('teste1')
    if selectbox == 'option2':
        st.markdown('teste2')

    multi = st.multiselect('Escolha as opções', ('option1', 'option2'))
    if multi == 'option1':
        st.markdown('teste1')
    if multi == 'option2':
        st.markdown('teste2')

    st.markdown('file uploader')
    file = st.file_uploader('Escolha seu arquivo', type='csv')
    if file is not None:
        st.markdown('Não esta vazio')
    else:
        st.markdown('Vazio')

if __name__ == '__main__':
    main()