"""
IA Generativa para leitura de PDF/Planilha com armazenamento em banco de dados
--------------------------------------------------------------------------
Fluxo:
1. Usuário faz upload de um PDF ou planilha (xlsx/csv)
2. O sistema extrai o conteúdo (texto ou tabela)
3. Uma IA generativa (Claude API) processa o conteúdo:
   - PDF -> gera um resumo + extrai campos estruturados em JSON
   - Planilha -> gera uma análise/insight em linguagem natural
4. O resultado (bruto + gerado pela IA) é salvo em um banco SQLite
5. A interface mostra o histórico de documentos já processados
"""

import io
import json
import sqlite3
from datetime import datetime

import pandas as pd
import pdfplumber
import streamlit as st
from google import genai

# ----------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------
DB_PATH = "dados.db"
# "gemini-flash-lite-latest" é um alias que sempre aponta para o modelo
# Flash-Lite mais recente — é o que tem a cota gratuita mais generosa hoje.
MODEL = "gemini-flash-lite-latest"

st.set_page_config(page_title="IA Leitora de Documentos", page_icon="🧠")


# ----------------------------------------------------------------------
# BANCO DE DADOS
# ----------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_arquivo TEXT,
            tipo TEXT,
            conteudo_bruto TEXT,
            resultado_ia TEXT,
            criado_em TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def salvar_documento(nome, tipo, bruto, resultado_ia):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO documentos (nome_arquivo, tipo, conteudo_bruto, resultado_ia, criado_em) "
        "VALUES (?, ?, ?, ?, ?)",
        (nome, tipo, bruto, resultado_ia, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def listar_documentos():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, nome_arquivo, tipo, resultado_ia, criado_em "
        "FROM documentos ORDER BY id DESC",
        conn,
    )
    conn.close()
    return df


def deletar_documentos(ids: list[int]):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    marcadores = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM documentos WHERE id IN ({marcadores})", ids)
    conn.commit()
    conn.close()


def limpar_todos_documentos():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documentos")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# EXTRAÇÃO
# ----------------------------------------------------------------------
def extrair_texto_pdf(arquivo) -> str:
    texto = []
    with pdfplumber.open(arquivo) as pdf:
        for pagina in pdf.pages:
            conteudo = pagina.extract_text()
            if conteudo:
                texto.append(conteudo)
    return "\n".join(texto)


def extrair_planilha(arquivo, nome: str) -> pd.DataFrame:
    if nome.endswith(".csv"):
        return pd.read_csv(arquivo)
    return pd.read_excel(arquivo)


# ----------------------------------------------------------------------
# CAMADA GENERATIVA (Google Gemini - API gratuita)
# ----------------------------------------------------------------------
def chamar_ia(prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    resposta = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return resposta.text


def processar_pdf_com_ia(texto: str, api_key: str) -> str:
    prompt = f"""Você recebeu o texto extraído de um PDF. Faça:
1. Um resumo de até 5 linhas.
2. Uma lista de possíveis campos estruturados encontrados (datas, valores, nomes, CNPJ, etc.), em formato JSON.

Texto do PDF:
\"\"\"{texto[:6000]}\"\"\"

Responda em português, formatado em markdown com as duas seções."""
    return chamar_ia(prompt, api_key)


def processar_planilha_com_ia(df: pd.DataFrame, api_key: str) -> str:
    amostra = df.head(20).to_csv(index=False)
    prompt = f"""Você recebeu uma amostra de uma planilha (primeiras 20 linhas, formato CSV).
Colunas: {list(df.columns)}

Amostra:
{amostra}

Gere uma análise curta (até 6 linhas) destacando padrões, valores fora do comum ou insights relevantes.
Responda em português."""
    return chamar_ia(prompt, api_key)


# ----------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------
def main():
    init_db()

    st.title("🧠 IA Generativa para Documentos")
    st.caption("Upload de PDF ou planilha → extração → IA generativa → banco de dados")

    api_key = st.text_input("Chave da API Google Gemini (gratuita)", type="password")

    arquivo = st.file_uploader(
        "Envie um arquivo PDF ou planilha (xlsx/csv)",
        type=["pdf", "xlsx", "csv"],
    )

    if arquivo and st.button("Processar"):
        if not api_key:
            st.error("Informe a chave da API para continuar.")
            return

        nome = arquivo.name
        conteudo_bytes = arquivo.read()

        with st.spinner("Extraindo conteúdo..."):
            if nome.endswith(".pdf"):
                texto = extrair_texto_pdf(io.BytesIO(conteudo_bytes))
                tipo = "pdf"
                bruto_para_salvar = texto
            else:
                df = extrair_planilha(io.BytesIO(conteudo_bytes), nome)
                tipo = "planilha"
                bruto_para_salvar = df.to_csv(index=False)

        with st.spinner("Gerando análise com IA..."):
            if tipo == "pdf":
                resultado_ia = processar_pdf_com_ia(texto, api_key)
            else:
                resultado_ia = processar_planilha_com_ia(df, api_key)

        salvar_documento(nome, tipo, bruto_para_salvar, resultado_ia)

        st.success("Documento processado e salvo no banco de dados!")
        st.subheader("Resultado da IA")
        st.markdown(resultado_ia)

        if tipo == "planilha":
            st.subheader("Prévia dos dados")
            st.dataframe(df.head(20))

    st.divider()
    st.subheader("📚 Histórico de documentos processados")
    historico = listar_documentos()

    if historico.empty:
        st.info("Nenhum documento processado ainda.")
        return

    st.dataframe(historico, use_container_width=True)

    opcoes = {
        f"#{row.id} — {row.nome_arquivo} ({row.criado_em[:16]})": row.id
        for row in historico.itertuples()
    }
    selecionados = st.multiselect(
        "Selecione os documentos que deseja excluir",
        options=list(opcoes.keys()),
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Excluir selecionados", disabled=not selecionados):
            ids_para_excluir = [opcoes[nome] for nome in selecionados]
            deletar_documentos(ids_para_excluir)
            st.success(f"{len(ids_para_excluir)} documento(s) excluído(s).")
            st.rerun()

    with col2:
        confirmar = st.checkbox("Confirmo que quero apagar TODO o histórico")
        if st.button("⚠️ Apagar todo o histórico", disabled=not confirmar):
            limpar_todos_documentos()
            st.success("Histórico apagado.")
            st.rerun()


if __name__ == "__main__":
    main()
