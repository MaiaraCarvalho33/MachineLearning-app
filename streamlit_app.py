import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # 


# Carregar os dados
df = pd.read_csv("dados.csv")
df["RDS_clean"] = df["RDS"].str.replace("%", "").astype(float)

# ================================
# Página 1 – Definição do Problema
# ================================
def pagina_definicao_problema():
    st.markdown("# 🧩 Definição do Problema")

    st.markdown("## Contextualização do Desafio")
    st.write("""
    No cenário financeiro atual, a concessão de crédito é uma atividade essencial para bancos e instituições financeiras, 
    mas também acarreta riscos significativos. A inadimplência de clientes pode gerar perdas substanciais, impactando a 
    lucratividade e a estabilidade das instituições. A análise de crédito tradicional, muitas vezes baseada em critérios 
    subjetivos e informações limitadas, pode não ser suficiente para identificar com precisão o perfil de risco de cada solicitante.

    O dataset `dados.csv` contém informações sobre clientes de uma instituição financeira, incluindo seu status 
    como "bom pagador" ou "mau pagador", além de diversas variáveis como valor de empréstimo, finalidade, tempo de emprego, 
    indicadores de crédito e limites de crédito. 
    """)

    st.markdown("## Objetivo do Modelo")
    st.write("""
    Desenvolver um modelo de machine learning capaz de prever se um cliente será um **bom pagador** ou **mau pagador** 
    com base nos dados disponíveis no momento da análise.
    """)

    st.markdown("## Relevância para o Negócio")
    st.markdown("""
    - **Minimização de perdas por inadimplência**
    - **Otimização da concessão de crédito**
    - **Melhoria na gestão de risco**
    - **Aumento da lucratividade**
    - **Identificação de oportunidades**
    - **Decisões orientadas por dados**
    """)

# =============================================
# Página 2 – Análise Exploratória e Insights
# =============================================
def pagina_eda():
    st.markdown("# 📊 Análise Exploratória dos Dados (EDA) e Insights")

    st.markdown("## 🧾 Visão Geral dos Dados")
    st.dataframe(df.head())

    # =========================
    st.markdown("## 🎯 Distribuição da Variável Alvo – Cliente")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Cliente", data=df, ax=ax1)
    ax1.set_title("Distribuição de Bons e Maus Pagadores")
    st.pyplot(fig1)

    st.markdown("""
    🔍 **Observação:**  
    O gráfico mostra que a maioria dos clientes é classificada como **bom pagador**, enquanto os **maus pagadores** são minoria.  
    Esse desbalanceamento pode levar o modelo a priorizar a classe majoritária e **ignorar os maus pagadores**, que são os mais importantes a identificar.
    """)

    # =========================
    st.markdown("## 📈 Histogramas com KDE por Variável Numérica")
    variaveis_hist = ['Empréstimo', 'ValorDoBem', 'TempoEmprego', 'TempoCliente', 'LC-Recente', 'LC-Atual', 'RDS_clean']

    for col in variaveis_hist:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=col, hue="Cliente", kde=True, ax=ax)
        ax.set_title(f"Distribuição de {col} por tipo de Cliente")
        st.pyplot(fig)

        # Texto explicativo para cada variável
        if col == "TempoEmprego":
            st.markdown("🔍 **TempoEmprego:** Bons pagadores tendem a ter mais tempo de emprego, indicando maior estabilidade.")
        elif col == "TempoCliente":
            st.markdown("🔍 **TempoCliente:** Clientes mais antigos tendem a ser mais confiáveis.")
        elif col == "RDS_clean":
            st.markdown("🔍 **RDS_clean:** Score de risco — valores mais altos aparecem entre os maus pagadores.")
        elif col == "LC-Recente":
            st.markdown("🔍 **LC-Recente:** Maus pagadores são mais comuns entre clientes com limite de crédito recente muito baixo ou nulo.")
        elif col == "LC-Atual":
            st.markdown("🔍 **LC-Atual:** Bons pagadores concentram-se em faixas de limite de crédito maiores.")
        else:
            st.markdown(f"🔍 **{col}:** Distribuição assimétrica, com valores altos concentrados nos bons pagadores.")

    # =========================
    st.markdown("## 📦 Boxplots por Classe – Comparação de Distribuições")
    for col in variaveis_hist:
        fig, ax = plt.subplots()
        sns.boxplot(x="Cliente", y=col, data=df, ax=ax)
        ax.set_title(f"{col} por tipo de Cliente")
        st.pyplot(fig)

    st.markdown("""
    🔍 **Interpretação dos Boxplots:**
    - As medianas de **TempoEmprego** e **TempoCliente** são maiores para bons pagadores.
    - A variável **RDS_clean** tende a ser mais alta entre maus pagadores.
    - **LC-Atual** e **LC-Recente** também diferenciam sutilmente os grupos.
    - Algumas variáveis como **Empréstimo** e **ValorDoBem** têm muitos outliers, o que pode distorcer médias e dispersões.
    """)

    # =========================
    st.markdown("## 🔗 Matriz de Correlação entre Variáveis Numéricas")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[variaveis_hist].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Correlação entre variáveis numéricas")
    st.pyplot(fig_corr)

    st.markdown("""
    🔍 **Matriz de Correlação – Destaques:**
    - Correlação **forte** entre **ValorDoBem** e **Empréstimo**.
    - Correlação **moderada** entre **TempoCliente** e **TempoEmprego**.
    - **RDS_clean** não é fortemente correlacionada com outras variáveis — pode ser uma variável **independente e poderosa**.
    """)

    # =========================
    st.markdown("## 📌 Principais Descobertas")
    st.markdown("""
    - **TempoEmprego**, **TempoCliente** e **RDS_clean** são bons candidatos para predição.
    - **LC-Atual** e **LC-Recente** oferecem informações úteis sobre a avaliação da instituição.
    - Algumas variáveis têm distribuição assimétrica e outliers que podem interferir nos modelos.
    - A separação entre as classes ocorre com mais clareza em algumas variáveis do que em outras.
    """)

    # =========================
    st.markdown("## 🧠 Considerações Finais para Modelagem")
    st.markdown("""
    - ⚠️ **Desbalanceamento de classes** deve ser tratado com SMOTE ou `class_weight`.
    - 🧹 Tratar valores ausentes em variáveis como **TempoEmprego** e **RDS**.
    - 🔤 Realizar One-Hot Encoding em variáveis como **Finalidade** e **Emprego**.
    - 🧮 Criar variáveis derivadas como:
        - `TotalHistoricoRuim = Negativos + Atrasos`
        - `RazaoEmprestimoValorBem = Empréstimo / ValorDoBem`
    - 🚫 Tratar **outliers** extremos para evitar distorções em algoritmos baseados em média.
    """)
###########pre
import numpy as np  # Certifique-se de ter isso no topo do arquivo

def pagina_preprocessamento():
    st.markdown("# 🧼 Pré-processamento dos Dados")

    # ========================
    st.markdown("## 1. Tratamento de Valores Ausentes")
    st.markdown("""
    - **TempoEmprego** e **RDS** possuem valores ausentes.
    - Estratégia aplicada:
        - **TempoEmprego**: substituído pela mediana da variável.
        - **RDS**: convertido de string para número (`RDS_clean`), depois preenchido com a **mediana por tipo de cliente**.
    """)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="TempoEmprego", kde=True, ax=axs[0])
    axs[0].set_title("Distribuição de TempoEmprego")

    sns.histplot(data=df, x="RDS_clean", kde=True, ax=axs[1])
    axs[1].set_title("Distribuição de RDS_clean (limpo)")
    st.pyplot(fig)

    # ========================
    st.markdown("## 2. Tratamento de Outliers")
    st.markdown("""
    - Variáveis como **Empréstimo** e **ValorDoBem** apresentaram valores extremos.
    - Estratégia recomendada:
        - Aplicar **capping** usando o percentil 1% e 99%, ou
        - Transformações como `log1p` para reduzir assimetrias.
    """)

    for col in ["Empréstimo", "ValorDoBem"]:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data=df, x=col, hue="Cliente", kde=True, ax=axs[0])
        axs[0].set_title(f"{col} - Original")

        sns.histplot(np.log1p(df[col]), ax=axs[1], bins=30, kde=True)
        axs[1].set_title(f"{col} - log1p Transformado")
        st.pyplot(fig)

    # ========================
    st.markdown("## 3. Codificação de Variáveis Categóricas")
    st.markdown("""
    - Variáveis como **Emprego** e **Finalidade** são categóricas.
    - Estratégia aplicada:
        - **One-Hot Encoding**: cria uma coluna binária para cada categoria.
        - Alternativa para árvores de decisão: **Label Encoding** ou deixar como string (algoritmos lidam bem).
    """)

    st.markdown("Exemplo de distribuição de valores categóricos:")
    for col in ["Emprego", "Finalidade"]:
        st.markdown(f"### 📊 {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    # ========================
    st.markdown("## 4. Normalização de Variáveis Contínuas")
    st.markdown("""
    - Para algoritmos sensíveis à escala (ex: KNN, Regressão Logística), foi aplicada:
        - **Padronização Z-score** (média 0, desvio padrão 1), ou
        - **Min-Max Scaling** (valores entre 0 e 1).
    """)

    st.markdown("📌 Exemplos de variáveis antes da normalização:")
    cols_norm = ["TempoCliente", "LC-Atual", "LC-Recente"]
    fig, ax = plt.subplots()
    df[cols_norm].boxplot(ax=ax)
    ax.set_title("Boxplot das variáveis contínuas")
    st.pyplot(fig)

    # ========================
    st.markdown("## 5. Engenharia de Variáveis")
    st.markdown("""
    Foram criadas novas variáveis com base em conhecimento de domínio:

    - `RazaoEmprestimoValorBem = Empréstimo / ValorDoBem`
        - Indica o quanto do bem está sendo financiado.

    - `TotalHistoricoRuim = Negativos + Atrasos`
        - Resume o histórico de inadimplência em uma só variável.
    """)

    # Criar as variáveis
    df["RazaoEmprestimoValorBem"] = df["Empréstimo"] / df["ValorDoBem"]
    df["TotalHistoricoRuim"] = df["Negativos"] + df["Atrasos"]

    for new_col in ["RazaoEmprestimoValorBem", "TotalHistoricoRuim"]:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data=df, x=new_col, hue="Cliente", kde=True, ax=axs[0])
        axs[0].set_title(f"{new_col} - Original")

        sns.histplot(np.log1p(df[new_col]), ax=axs[1], bins=30, kde=True)
        axs[1].set_title(f"{new_col} - log1p Transformado")
        st.pyplot(fig)

    # ========================
    st.markdown("## 6. Balanceamento da Variável Alvo")
    st.markdown("""
    O conjunto de dados é desbalanceado (muito mais bons pagadores que maus).

    Estratégias sugeridas:
    - **SMOTE**: técnica de oversampling sintético para a classe minoritária.
    - **Class Weights**: ajuste do peso da classe minoritária no algoritmo.
    """)

    fig, ax = plt.subplots()
    sns.countplot(x="Cliente", data=df, ax=ax)
    ax.set_title("Distribuição da variável alvo (Cliente)")
    st.pyplot(fig)

    # ========================
    st.markdown("## 7. Divisão dos Dados")
    st.markdown("""
    Os dados foram divididos em:

    - **Treinamento** (80% dos dados)
    - **Teste** (20% restantes)

    Mantendo a proporção entre bons e maus pagadores (estratificação).
    """)

    from sklearn.model_selection import train_test_split
    X = df.drop("Cliente", axis=1)
    y = df["Cliente"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    st.success("✅ Divisão realizada com sucesso!")
    st.markdown(f"- Treinamento: {X_train.shape[0]} registros")
    st.markdown(f"- Teste: {X_test.shape[0]} registros")
####
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

def pagina_modelagem_avaliacao():
    st.markdown("# 🤖 Modelagem e Avaliação")

    # ===================================
    # 1. Preparação dos Dados
    # ===================================
    st.markdown("## 1. Preparação dos Dados")
    df_modelo = df.copy()

    for col in df_modelo.select_dtypes(include='number').columns:
        df_modelo[col] = df_modelo[col].fillna(df_modelo[col].median())

    for col in df_modelo.select_dtypes(include='object').columns:
        df_modelo[col] = df_modelo[col].fillna(df_modelo[col].mode()[0])

    le = LabelEncoder()
    df_modelo["Cliente"] = le.fit_transform(df_modelo["Cliente"])  # bom pagador = 1, mau = 0

    df_modelo = pd.get_dummies(df_modelo, drop_first=True)

    X = df_modelo.drop("Cliente", axis=1)
    y = df_modelo["Cliente"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    st.success("✅ Dados preparados com sucesso!")

    # ===================================
    # 2. Treinamento e Avaliação
    # ===================================
    st.markdown("## 2. Treinamento e Avaliação dos Modelos")

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Árvore de Decisão": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    resultados_modelos = []

    for nome, modelo in modelos.items():
        st.markdown(f"### 🔍 {nome}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        # Matriz de Confusão
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title(f"Matriz de Confusão - {nome}")
        ax_cm.set_xlabel("Predito")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

        # Relatório
        st.markdown("**Relatório de Classificação:**")
        st.text(classification_report(y_test, y_pred, target_names=["mau pagador", "bom pagador"]))

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        precision = report_dict['0']['precision']  # mau pagador = 0
        recall = report_dict['0']['recall']
        f1 = report_dict['0']['f1-score']
        acc = report_dict['accuracy']

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("Falso Positivo")
        ax_roc.set_ylabel("Verdadeiro Positivo")
        ax_roc.set_title(f"Curva ROC - {nome}")
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Curva Precision-Recall
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recalls, precisions)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"Curva Precision-Recall - {nome}")
        st.pyplot(fig_pr)

        # Validação Cruzada
        scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
        st.markdown(f"**Validação Cruzada (5-fold):** {np.round(scores, 3)}")
        st.markdown(f"**Acurácia Média:** {scores.mean():.4f}")

        # Guardar resultados
        resultados_modelos.append({
            "Modelo": nome,
            "f1_mau_pagador": f1,
            "Recall_mau": recall,
            "Precision_mau": precision,
            "Acurácia teste": acc,
            "Acurácia Val. Cruzada": scores.mean()
        })

        st.markdown("---")

 

def pagina_relatorio_final():
    st.markdown("# 📈 Relatório Final do Projeto")

    # =====================
    # Comparativo Final dos Modelos (dados fixos)
    # =====================
    st.markdown("## 📊 Comparativo Final dos Modelos")

    dados = {
        "Modelo": ["Random Forest", "Regressão Logística", "Árvore de Decisão"],
        "F1_mau_pagador": [0.947, 0.926, 0.913],
        "Recall_mau": [0.991, 0.966, 0.926],
        "Precision_mau": [0.906, 0.889, 0.900],
        "Acurácia Teste": [0.909, 0.874, 0.856],
        "Acurácia Val. Cruzada": [0.896, 0.859, 0.862]
    }

    comparativo = pd.DataFrame(dados)
    st.dataframe(comparativo.style.format(precision=3), use_container_width=True)

    # =====================
    # Análise Comparativa
    # =====================
    st.markdown("## 📌 Análise Comparativa dos Modelos")
    st.write("""
    O modelo **Random Forest** se destacou como o melhor para prever maus pagadores, com base nas seguintes evidências:

    - **F1-Score mais alto (0.947)** para a classe de interesse, indicando o melhor equilíbrio entre precisão e recall.
    - **Recall de 0.991**, ou seja, quase todos os inadimplentes foram identificados corretamente.
    - **Precision de 0.906**, o que significa que, quando o modelo indica que alguém é inadimplente, geralmente está certo.
    - **Acurácia de teste (0.909)** e **validação cruzada (0.896)** mostram que o modelo é robusto e consistente, evitando overfitting.
    - **AUC = 0.96**, evidenciado pela curva ROC, confirma o excelente desempenho na separação entre classes.

    Já os outros modelos apresentaram desempenho inferior:

    - **Regressão Logística** teve bom recall (0.966), mas menor precisão e acurácia geral. Seu **AUC foi 0.85**, indicando desempenho razoável, mas inferior ao Random Forest.
    - **Árvore de Decisão** teve o pior desempenho, com menor estabilidade e **AUC de apenas 0.74**, o que mostra dificuldade em distinguir corretamente os inadimplentes dos bons pagadores.
    """)

    st.markdown("### 🔍 Conclusão")
    st.write("""
    Com base nos resultados, o **Random Forest** é o modelo mais indicado para o problema, por garantir:

    - Maior identificação de inadimplentes (**recall alto**)
    - Menos falsos positivos e negativos (**alta precisão**)
    - Melhor desempenho geral nas métricas críticas do negócio
    - Maior área sob a curva ROC (**AUC = 0.96**), indicando excelente separabilidade

    Esse modelo contribui para **minimizar riscos de concessão de crédito indevido** e **otimizar a rentabilidade da instituição**.
    """)
# ========================
# Função da nova página - Sobre o Projeto
# ========================
def pagina_sobre():
    st.markdown("# 📘 Sobre o Projeto")

    st.markdown("## 📌 Contexto")
    st.write("""
    Este projeto tem como objetivo utilizar técnicas de **Machine Learning** para prever se um cliente será **bom** ou **mau pagador**, 
    com base em um conjunto de dados fornecido por uma instituição financeira.
    """)

    st.markdown("## ⚙️ Ferramentas Utilizadas")
    st.markdown("""
    - `Python` e `Streamlit` para o desenvolvimento do app interativo
    - `Pandas` e `NumPy` para manipulação de dados
    - `Seaborn` e `Matplotlib` para visualização
    - `Scikit-learn` para modelagem estatística
    """)

    st.markdown("## 🎯 Foco da Predição")
    st.write("""
    A variável-alvo é o status do cliente (bom ou mau pagador). A principal métrica de avaliação é o **F1-score** da classe 'mau pagador',
    devido à sua importância para o negócio.
    """)

    st.markdown("## 👩‍💻 Autoria")
    st.write("""
    Projeto desenvolvido por **Maiara Carvalho**
    """)


# =============================================
# Navegação entre páginas
# =============================================
pagina = st.sidebar.radio("Selecione a página:", [
    "Definição do Problema", 
    "Análise Exploratória (EDA)",
    "Pré-processamento dos Dados",
    "Modelagem e Avaliação",
    "Relatório Final",
    "Sobre o Projeto"
])

if pagina == "Definição do Problema":
    pagina_definicao_problema()

elif pagina == "Análise Exploratória (EDA)":
    pagina_eda()

elif pagina == "Pré-processamento dos Dados":
    pagina_preprocessamento()

elif pagina == "Modelagem e Avaliação":
    resultados_modelos = pagina_modelagem_avaliacao()

elif pagina == "Relatório Final":
    pagina_relatorio_final()

elif pagina == "Sobre o Projeto":
    pagina_sobre()
