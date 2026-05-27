import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dasbor Analisis Kinerja Siswa", layout="wide")

st.title("Dasbor Analisis Kinerja Siswa")
st.markdown("""
Dasbor ini menganalisis kinerja siswa berdasarkan berbagai faktor termasuk kebiasaan belajar,
gaya hidup, dan karakteristik pribadi.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('student_habits_performance.csv')
    return df

try:
    df = load_data()

    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pergi ke", [
        "1. Pemahaman Data",
        "2. Pra-pemrosesan Data",
        "3. Analisis Eksplorasi",
        "4. Rekayasa Fitur",
        "5. Analisis Klaster",
        "6. Pemodelan & Evaluasi",
        "7. Prediksi"
    ])

    if page == "1. Pemahaman Data":
        st.header("Pemahaman Data")
        st.write("Bentuk dataset:", df.shape)

        st.subheader("Contoh Data")
        st.dataframe(df.head())

        st.subheader("Informasi Data")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Statistik Data Numerik")
        st.write(df.describe())

        categorical_cols = ['part_time_job', 'diet_quality',
                          'parental_education_level', 'internet_quality',
                          'extracurricular_participation']

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            fig = px.pie(values=value_counts.values,
                        names=value_counts.index,
                        title=f'Distribusi {col.replace("_", " ").title()}')
            st.plotly_chart(fig)

        st.subheader("Analisis Nilai Hilang")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Jumlah Nilai Hilang:")
            st.write(missing_values[missing_values > 0])

            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
            plt.title('Peta Panah Nilai Hilang')
            st.pyplot(fig)
        else:
            st.success("Tidak ditemukan nilai hilang dalam dataset!")

    elif page == "2. Pra-pemrosesan Data":
        st.header("Pra-pemrosesan Data")

        st.subheader("Ringkasan Data Asli")
        st.write("Bentuk:", df.shape)

        st.subheader("1. Penanganan Nilai Hilang")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Nilai hilang ditemukan:")
            st.write(missing_values[missing_values > 0])
        else:
            st.success("Tidak ada nilai hilang yang perlu ditangani!")

        st.subheader("2. Deteksi Pencilan")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        selected_col = st.selectbox("Pilih kolom untuk analisis pencilan:", numeric_cols)

        fig = go.Figure()
        fig.add_trace(go.Box(y=df[selected_col], name=selected_col))
        fig.update_layout(title=f'Box Plot {selected_col}')
        st.plotly_chart(fig)

        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[selected_col] < (Q1 - 1.5 * IQR)) | (df[selected_col] > (Q3 + 1.5 * IQR))][selected_col]
        st.write(f"Jumlah pencilan terdeteksi: {len(outliers)}")

        st.subheader("3. Pratinjau Skala Fitur")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistik Data Asli:")
            st.write(df[numeric_cols].describe())
        with col2:
            st.write("Statistik Data Terskala:")
            st.write(scaled_df.describe())

    elif page == "3. Analisis Eksplorasi":
        st.header("Analisis Data Eksplorasi")

        tab1, tab2, tab3 = st.tabs(["Analisis Univariate", "Analisis Bivariate", "Analisis Multivariate"])

        with tab1:
            st.subheader("Analisis Univariate")

            st.write("### Variabel Numerik")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            selected_num = st.selectbox("Pilih variabel numerik:", numeric_cols)

            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Plot Distribusi', 'Box Plot'))

            fig.add_trace(go.Histogram(x=df[selected_num], name="Distribusi"),
                         row=1, col=1)
            fig.add_trace(go.Box(x=df[selected_num], name="Box Plot"),
                         row=2, col=1)

            fig.update_layout(height=600, title_text=f"Analisis {selected_num}")
            st.plotly_chart(fig)

            st.write("### Variabel Kategorikal")
            categorical_cols = ['part_time_job', 'diet_quality',
                              'parental_education_level', 'internet_quality',
                              'extracurricular_participation']

            selected_cat = st.selectbox("Pilih variabel kategorikal:", categorical_cols)

            value_counts = df[selected_cat].value_counts()
            percentages = (value_counts / len(df) * 100).round(1)

            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        text=[f'{p}%' for p in percentages],
                        title=f'Distribusi {selected_cat}')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig)

        with tab2:
            st.subheader("Analisis Bivariate")

            analysis_type = st.radio("Pilih Jenis Analisis:",
                                   ["Numerik vs Numerik",
                                    "Numerik vs Kategorikal",
                                    "Kategorikal vs Kategorikal"])

            if analysis_type == "Numerik vs Numerik":
                x_num = st.selectbox("Pilih variabel X:", numeric_cols, key='x_num')
                y_num = st.selectbox("Pilih variabel Y:", numeric_cols, key='y_num')

                fig = px.scatter(df, x=x_num, y=y_num,
                               trendline="ols",
                               title=f'{x_num} vs {y_num}')
                st.plotly_chart(fig)

                correlation = df[x_num].corr(df[y_num])
                st.write(f"Koefisien korelasi: {correlation:.2f}")

            elif analysis_type == "Numerik vs Kategorikal":
                num_var = st.selectbox("Pilih Variabel Numerik:", numeric_cols)
                cat_var = st.selectbox("Pilih Variabel Kategorikal:", categorical_cols)

                fig = px.box(df, x=cat_var, y=num_var,
                           title=f'{num_var} berdasarkan {cat_var}')
                st.plotly_chart(fig)

                categories = df[cat_var].unique()
                f_stat, p_val = f_oneway(*[df[df[cat_var] == cat][num_var]
                                         for cat in categories])
                st.write(f"Hasil uji ANOVA:")
                st.write(f"F-statistik: {f_stat:.2f}")
                st.write(f"p-value: {p_val:.4f}")

            else:
                cat_var1 = st.selectbox("Pilih Variabel Kategorikal Pertama:",
                                      categorical_cols, key='cat1')
                cat_var2 = st.selectbox("Pilih Variabel Kategorikal Kedua:",
                                      categorical_cols, key='cat2')

                contingency = pd.crosstab(df[cat_var1], df[cat_var2])
                fig = px.imshow(contingency,
                              title=f'Hubungan antara {cat_var1} dan {cat_var2}')
                st.plotly_chart(fig)

                chi2, p_val, dof, expected = chi2_contingency(contingency)
                st.write(f"Hasil uji Chi-square:")
                st.write(f"Statistik Chi-square: {chi2:.2f}")
                st.write(f"p-value: {p_val:.4f}")

        with tab3:
            st.subheader("Analisis Multivariate")

            st.write("### Matriks Korelasi")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr,
                          title='Matriks Korelasi',
                          labels=dict(color="Korelasi"))
            st.plotly_chart(fig)

            st.write("### Analisis PCA")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            exp_var_ratio = pca.explained_variance_ratio_
            cum_exp_var_ratio = np.cumsum(exp_var_ratio)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(1, len(exp_var_ratio) + 1)),
                                y=exp_var_ratio,
                                name='Individual'))
            fig.add_trace(go.Scatter(x=list(range(1, len(cum_exp_var_ratio) + 1)),
                                   y=cum_exp_var_ratio,
                                   name='Kumulatif'))
            fig.update_layout(title='Rasio Varians yang Dijelaskan',
                            xaxis_title='Komponen Utama',
                            yaxis_title='Rasio Varians yang Dijelaskan')
            st.plotly_chart(fig)

    elif page == "4. Rekayasa Fitur":
        st.header("Rekayasa Fitur")

        st.subheader("1. Skala Fitur")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

        selected_col = st.selectbox("Pilih kolom untuk melihat efek skala:", numeric_cols)

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Data Asli', 'Data Terskala'))

        fig.add_trace(go.Histogram(x=df[selected_col], name="Asli"),
                     row=1, col=1)
        fig.add_trace(go.Histogram(x=scaled_df[selected_col], name="Terskala"),
                     row=1, col=2)

        fig.update_layout(title_text=f"Efek Skala pada {selected_col}")
        st.plotly_chart(fig)

        st.subheader("2. Transformasi Fitur")
        transform_type = st.selectbox("Pilih transformasi:",
                                    ["Standard Scaling"])

        selected_col = st.selectbox("Pilih kolom untuk ditransformasi:",
                                  numeric_cols, key='transform')

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Distribusi Asli',
                                         'Distribusi Ditransformasi'))

        fig.add_trace(go.Histogram(x=df[selected_col],
                                 name="Asli"), row=1, col=1)

        scaler = StandardScaler()
        transformed = scaler.fit_transform(df[[selected_col]]).flatten()

        fig.add_trace(go.Histogram(x=transformed,
                                 name="Ditransformasi"), row=1, col=2)

        fig.update_layout(title_text="Efek Standard Scaling")
        st.plotly_chart(fig)

        # Menampilkan statistik sebelum dan sesudah transformasi
        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistik Data Asli:")
            st.write(pd.DataFrame({
                'Min': [df[selected_col].min()],
                'Max': [df[selected_col].max()],
                'Mean': [df[selected_col].mean()],
                'Std': [df[selected_col].std()]
            }))
        with col2:
            st.write("Statistik Data Ditransformasi:")
            st.write(pd.DataFrame({
                'Min': [transformed.min()],
                'Max': [transformed.max()],
                'Mean': [transformed.mean()],
                'Std': [transformed.std()]
            }))

        st.subheader("3. Enkode Fitur")
        categorical_cols = ['part_time_job', 'diet_quality',
                          'parental_education_level', 'internet_quality',
                          'extracurricular_participation']

        selected_cat = st.selectbox("Pilih fitur kategorikal:", categorical_cols)

        le = LabelEncoder()
        ohe = OneHotEncoder(sparse_output=False)

        label_encoded = le.fit_transform(df[selected_cat])
        onehot_encoded = ohe.fit_transform(df[[selected_cat]])

        col1, col2 = st.columns(2)

        with col1:
            st.write("Label Encoding:")
            encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
            st.write(encoding_map)

        with col2:
            st.write("One-Hot Encoding:")
            onehot_df = pd.DataFrame(onehot_encoded,
                                   columns=ohe.get_feature_names_out([selected_cat]))
            st.write(onehot_df.head())

    elif page == "5. Analisis Klaster":
        st.header("Analisis Klaster Siswa")

        cluster_features = ['age', 'study_hours_per_day', 'social_media_hours',
                  'netflix_hours', 'attendance_percentage', 'sleep_hours',
                  'exercise_frequency', 'mental_health_rating']

        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(df[cluster_features])

        joblib.dump(scaler, 'cluster_scaler.joblib')

        st.subheader("Jumlah Klaster Optimal")
        max_clusters = 10
        inertias = []

        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_cluster)
            inertias.append(kmeans.inertia_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, max_clusters + 1)),
                                y=inertias,
                                mode='lines+markers'))
        fig.update_layout(title='Metode Elbow untuk k Optimal',
                         xaxis_title='Jumlah Klaster (k)',
                         yaxis_title='Inersia')
        st.plotly_chart(fig)

        n_clusters = st.slider("Pilih jumlah klaster:", 2, 10, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster)

        joblib.dump(kmeans, 'kmeans_model.joblib')

        df_cluster = df.copy()
        df_cluster['Cluster'] = cluster_labels

        st.subheader("Analisis Klaster")

        cluster_sizes = df_cluster['Cluster'].value_counts().sort_index()
        st.write("Ukuran Klaster:")
        st.write(cluster_sizes)

        st.subheader("Karakteristik Klaster")

        cluster_means = df_cluster.groupby('Cluster')[cluster_features].mean()

        for feature in cluster_features:
            fig = px.box(df_cluster, x='Cluster', y=feature,
                        title=f'Distribusi {feature} per Klaster')
            st.plotly_chart(fig)

        st.subheader("Visualisasi 2D Klaster")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster)

        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = cluster_labels

        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                        title='Visualisasi Klaster menggunakan PCA')
        st.plotly_chart(fig)

        st.subheader("Profil Klaster")

        for cluster in range(n_clusters):
            st.write(f"\nProfil Klaster {cluster}:")
            cluster_data = cluster_means.loc[cluster]

            profile = []
            for feature in cluster_features:
                value = cluster_data[feature]
                overall_mean = df[feature].mean()

                if value > overall_mean:
                    level = "Lebih tinggi"
                else:
                    level = "Lebih rendah"

                profile.append(f"- {feature.replace('_', ' ').title()}: {level} dari rata-rata ({value:.2f} vs {overall_mean:.2f})")

            for point in profile:
                st.write(point)

    elif page == "6. Pemodelan & Evaluasi":
        st.header("Pemodelan & Evaluasi")

        numeric_features = ['age', 'study_hours_per_day', 'social_media_hours',
                          'netflix_hours', 'attendance_percentage', 'sleep_hours',
                          'exercise_frequency', 'mental_health_rating']
        categorical_features = ['part_time_job', 'diet_quality',
                              'parental_education_level', 'internet_quality',
                              'extracurricular_participation']

        df['parental_education_level'] = df['parental_education_level'].replace('None', 'High School')

        st.subheader("Langkah 1: Klasterisasi")
        X_cluster = StandardScaler().fit_transform(df[numeric_features])
        n_clusters = st.slider("Pilih jumlah klaster untuk pemodelan:", 2, 5, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster)

        joblib.dump(kmeans, 'kmeans_model.joblib')

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels

        st.write("Distribusi klaster:")
        cluster_counts = pd.DataFrame(df_with_clusters['cluster'].value_counts()).reset_index()
        cluster_counts.columns = ['Klaster', 'Jumlah']
        st.write(cluster_counts)

        st.subheader("Langkah 2: Pemodelan Berbasis Klaster")

        def label_exam(score):
            if score >= 85:
                return "A"
            elif score >= 75:
                return "B"
            elif score >= 65:
                return "C"
            elif score >= 50:
                return "D"
            else:
                return "E"

        df_with_clusters['exam_label'] = df_with_clusters['exam_score'].apply(label_exam)

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        for cluster in range(n_clusters):
            st.write(f"\nModel Klaster {cluster}:")

            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]

            if len(cluster_data) < 2:
                st.warning(f"Data tidak cukup di klaster {cluster} untuk pemodelan")
                continue

            X = cluster_data[numeric_features + categorical_features]
            y = cluster_data['exam_label']

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                   test_size=0.2,
                                                                   random_state=42)
            except ValueError as e:
                st.warning(f"Sampel tidak cukup di klaster {cluster}: {str(e)}")
                continue

            model = GaussianNB()

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)

            joblib.dump(pipeline, f'model_pipeline_cluster_{cluster}.joblib')

            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.metric("Akurasi", f"{acc:.2f}")

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm,
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=sorted(y.unique()),
                           y=sorted(y.unique()),
                           title=f'Confusion Matrix - Klaster {cluster}')
            st.plotly_chart(fig)

    elif page == "7. Prediksi":
        st.header("Prediksi Kinerja Siswa")

        try:
            kmeans = joblib.load('kmeans_model.joblib')
            cluster_scaler = joblib.load('cluster_scaler.joblib')
        except:
            st.error("Model klaster atau scaler tidak ditemukan. Silakan latih model di bagian 'Pemodelan & Evaluasi' terlebih dahulu.")
            st.stop()

        st.subheader("Masukkan Informasi Siswa")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Fitur Numerik:")
            age = st.number_input("Usia", 18, 30, 20)
            study_hours = st.number_input("Jam Belajar per Hari", 0.0, 24.0, 5.0)
            social_media = st.number_input("Jam Media Sosial", 0.0, 24.0, 2.0)
            netflix = st.number_input("Jam Netflix", 0.0, 24.0, 1.0)
            attendance = st.number_input("Persentase Kehadiran", 0.0, 100.0, 85.0)
            sleep = st.number_input("Jam Tidur", 0.0, 24.0, 7.0)
            exercise = st.number_input("Frekuensi Olahraga (hari per minggu)", 0, 7, 3)
            mental_health = st.slider("Penilaian Kesehatan Mental", 1, 10, 5)

        with col2:
            st.write("Fitur Kategorikal:")
            part_time_job = st.selectbox("Pekerjaan Paruh Waktu", ['Yes', 'No'])
            diet_quality = st.selectbox("Kualitas Diet", ['Poor', 'Average', 'Good'])
            parental_education = st.selectbox("Pendidikan Orang Tua",
                                            ['High School', 'Bachelor', 'Master'])
            internet_quality = st.selectbox("Kualitas Internet",
                                          ['Poor', 'Average', 'Good'])
            extracurricular = st.selectbox("Partisipasi Ekstrakurikuler",
                                         ['Yes', 'No'])

        if st.button("Prediksi Kinerja"):
            input_data = pd.DataFrame({
                'age': [age],
                'study_hours_per_day': [study_hours],
                'social_media_hours': [social_media],
                'netflix_hours': [netflix],
                'attendance_percentage': [attendance],
                'sleep_hours': [sleep],
                'exercise_frequency': [exercise],
                'mental_health_rating': [mental_health],
                'part_time_job': [part_time_job],
                'diet_quality': [diet_quality],
                'parental_education_level': [parental_education],
                'internet_quality': [internet_quality],
                'extracurricular_participation': [extracurricular]
            })

            cluster_features = ['age', 'study_hours_per_day', 'social_media_hours',
                              'netflix_hours', 'attendance_percentage', 'sleep_hours',
                              'exercise_frequency', 'mental_health_rating']

            X_cluster = cluster_scaler.transform(input_data[cluster_features])
            cluster = kmeans.predict(X_cluster)[0]

            try:
                pipeline = joblib.load(f'model_pipeline_cluster_{cluster}.joblib')
                prediction = pipeline.predict(input_data)[0]

                st.success(f"Siswa termasuk dalam Klaster {cluster}")
                st.success(f"Prediksi Nilai: {prediction}")

                st.subheader("Analisis Kinerja & Rekomendasi")

                analysis = []

                if study_hours < 6:
                    analysis.append({
                        'Faktor': 'Jam Belajar',
                        'Status': 'Perlu Peningkatan',
                        'Saat Ini': f'{study_hours:.1f} jam',
                        'Rekomendasi': 'Tingkatkan waktu belajar minimal 6 jam per hari'
                    })

                if sleep < 7:
                    analysis.append({
                        'Faktor': 'Tidur',
                        'Status': 'Perlu Peningkatan',
                        'Saat Ini': f'{sleep:.1f} jam',
                        'Rekomendasi': 'Coba tidur minimal 7 jam'
                    })

                if exercise < 3:
                    analysis.append({
                        'Faktor': 'Olahraga',
                        'Status': 'Perlu Peningkatan',
                        'Saat Ini': f'{exercise} hari/minggu',
                        'Rekomendasi': 'Olahraga minimal 3 hari per minggu'
                    })

                if mental_health < 7:
                    analysis.append({
                        'Faktor': 'Kesehatan Mental',
                        'Status': 'Perlu Perhatian',
                        'Saat Ini': f'{mental_health}/10',
                        'Rekomendasi': 'Pertimbangkan mencari dukungan atau konseling'
                    })

                if social_media > 3:
                    analysis.append({
                        'Faktor': 'Media Sosial',
                        'Status': 'Perlu Perhatian',
                        'Saat Ini': f'{social_media:.1f} jam',
                        'Rekomendasi': 'Coba kurangi penggunaan media sosial'
                    })

                if attendance < 80:
                    analysis.append({
                        'Faktor': 'Kehadiran',
                        'Status': 'Perlu Peningkatan',
                        'Saat Ini': f'{attendance:.1f}%',
                        'Rekomendasi': 'Jaga kehadiran di atas 80%'
                    })

                if diet_quality == 'Poor':
                    analysis.append({
                        'Faktor': 'Kualitas Diet',
                        'Status': 'Perlu Peningkatan',
                        'Saat Ini': diet_quality,
                        'Rekomendasi': 'Tingkatkan kualitas diet untuk kinerja lebih baik'
                    })

                if analysis:
                    analysis_df = pd.DataFrame(analysis)
                    st.table(analysis_df)
                else:
                    st.success("Bagus! Kebiasaan saat ini sudah seimbang.")

            except Exception as e:
                st.error(f"Error memuat model untuk klaster {cluster}: {str(e)}")
                st.write("Pastikan model telah dilatih di bagian 'Pemodelan & Evaluasi'.")

except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.write("Pastikan file data berada di lokasi dan format yang benar.")
