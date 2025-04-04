import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import os
import tempfile

# Check if summarytools is available
summarytools_available = importlib.util.find_spec("summarytools") is not None

# Set page configuration
st.set_page_config(page_title="Analytical Toolkit", layout="wide")

# App title and description
st.title("Analytical Toolkit")
st.markdown("""
This app allows you to upload an Excel file and perform various exploratory data analysis tasks:
* Create **pairplots** by selecting specific columns to visualize relationships
* Generate a **correlation matrix** for numerical columns
* Produce **Q-Q plots** to check for normality in your data

Simply drag and drop your Excel file below to get started.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

# Function to get only numeric columns
def get_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

# Main app logic
if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Display basic dataframe info
        st.write(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Show the first few rows of the dataframe
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Create tabs for different analyses
        tab0, tab1, tab2, tab3, tab4 = st.tabs(["Data Summary", "Pairplot", "Correlation Matrix", "Q-Q Plot", "Parallel Coordinates"])
        
        # Tab 0: Data Summary
        with tab0:
            st.subheader("Data Summary")
            st.markdown("Comprehensive summary of the dataset with statistics and information about each column")
            
            # Try to use summarytools if available
            if summarytools_available:
                try:
                    from summarytools import dfSummary
                    
                    # Add an option to install summarytools if not available
                    st.info("""
                    Trying to use the summarytools library. If you're seeing this message for a long time, 
                    the library might not be properly configured.
                    """)
                    
                    # Create a temporary HTML file to store the dfSummary output
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                        # Generate the dfSummary output
                        summary = dfSummary(df)
                        # Save to the temporary file
                        summary.to_html(tmp_file.name)
                    
                    # Read the HTML file
                    with open(tmp_file.name, 'r') as f:
                        html_string = f.read()
                    
                    # Clean up the file
                    os.unlink(tmp_file.name)
                    
                    # Display the HTML
                    st.components.v1.html(html_string, height=1024, scrolling=True)
                    
                    st.success("Successfully used summarytools.dfSummary for the data summary!")
                except Exception as e:
                    st.error(f"Error generating dfSummary from summarytools: {e}")
                    # Switch to custom summary
                    show_custom_summary = True
            else:
                st.warning("""
                The summarytools library is not available in this environment. 
                Using a custom data summary implementation instead.
                
                If you want to use summarytools, you can install it with:
                ```
                pip install summarytools
                ```
                """)
                show_custom_summary = True
        
        
        # Tab 1: Pairplot
        with tab1:
            st.subheader("Create Pairplot")
            st.markdown("Select columns to visualize relationships between variables")
            
            # Get all column names
            all_columns = df.columns.tolist()
            
            # Let the user select columns for the pairplot
            selected_columns = st.multiselect(
                "Select columns for pairplot (2-6 columns recommended for readability)",
                all_columns,
                default=get_numeric_columns(df)[:4] if len(get_numeric_columns(df)) >= 4 else get_numeric_columns(df)
            )
            
            # Option to select a hue column
            hue_column = st.selectbox(
                "Select a categorical column for hue (optional)",
                ["None"] + [col for col in all_columns if df[col].nunique() < 10 and col not in selected_columns],
                index=0
            )
            
            if st.button("Generate Pairplot") and len(selected_columns) >= 2:
                st.write("Generating pairplot... this may take a moment.")
                
                # Create a custom pairplot with transparent distribution on diagonal, similar to Seaborn style
                n = len(selected_columns)
                fig = make_subplots(rows=n, cols=n, shared_xaxes=False, shared_yaxes=False,
                                   horizontal_spacing=0.05, vertical_spacing=0.05)
                
                # Create color mapping if hue is selected - use colors more similar to Seaborn's default palette
                if hue_column != "None":
                    hue_categories = df[hue_column].unique()
                    # Use more Seaborn-like colors
                    custom_colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3']
                    hue_color_map = {cat: custom_colors[i % len(custom_colors)] for i, cat in enumerate(hue_categories)}
                    
                    # Prepare marker symbols for different categories (circles, squares, diamonds)
                    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x']
                    hue_symbol_map = {cat: marker_symbols[i % len(marker_symbols)] for i, cat in enumerate(hue_categories)}
                
                # Generate all subplots
                for i, col_i in enumerate(selected_columns):
                    for j, col_j in enumerate(selected_columns):
                        if i == j:  # Diagonal - create density plot
                            # Create KDE for the diagonal
                            kde_data = df[col_i].dropna()
                            
                            if len(kde_data) > 1:  # Ensure we have enough data for KDE
                                # If hue is selected, create multiple KDEs
                                if hue_column != "None":
                                    for hue_val in hue_categories:
                                        hue_kde_data = df[df[hue_column] == hue_val][col_i].dropna()
                                        if len(hue_kde_data) > 1:
                                            # Calculate the bandwidth to get a smoother curve
                                            bw = 'scott'  # Seaborn-like bandwidth estimation
                                            x_hue = np.linspace(hue_kde_data.min(), hue_kde_data.max(), 1000)
                                            try:
                                                kde_hue = stats.gaussian_kde(hue_kde_data, bw_method=bw)
                                                y_hue = kde_hue(x_hue)
                                                
                                                # Convert color hex to rgba for fill transparency
                                                color = hue_color_map[hue_val]
                                                r = int(color[1:3], 16)
                                                g = int(color[3:5], 16)
                                                b = int(color[5:7], 16)
                                                
                                                fig.add_trace(
                                                    go.Scatter(
                                                        x=x_hue, 
                                                        y=y_hue,
                                                        mode='lines',
                                                        fill='tozeroy',
                                                        fillcolor=f'rgba({r}, {g}, {b}, 0.3)',  # More transparent fill
                                                        line=dict(color=color, width=1.5),
                                                        name=f"{hue_val}",
                                                        showlegend=(i==0 and j==0),
                                                    ),
                                                    row=i+1, col=j+1
                                                )
                                            except np.linalg.LinAlgError:
                                                # Fall back to simpler approach if KDE fails
                                                pass
                                else:
                                    # For no hue, create a single blue density plot
                                    try:
                                        x_range = np.linspace(kde_data.min(), kde_data.max(), 1000)
                                        kde = stats.gaussian_kde(kde_data, bw_method='scott')
                                        y_range = kde(x_range)
                                    
                                        # Add the overall KDE with transparency for the diagonal
                                        fig.add_trace(
                                            go.Scatter(
                                                x=x_range, 
                                                y=y_range,
                                                mode='lines',
                                                fill='tozeroy',
                                                fillcolor='rgba(76, 114, 176, 0.3)',  # Seaborn-like blue, transparent
                                                line=dict(color='rgb(76, 114, 176)', width=1.5),
                                                name=col_i,
                                                showlegend=False
                                            ),
                                            row=i+1, col=j+1
                                        )
                                    except np.linalg.LinAlgError:
                                        # Fall back to simpler approach if KDE fails
                                        pass
                                
                                # Mantieni le etichette sull'asse y anche per i grafici sulla diagonale
                                # Non rimuoviamo i numeri nell'asse y
                                
                        else:  # Off-diagonal - create scatter plot
                            if hue_column != "None":
                                for hue_val in hue_categories:
                                    hue_data = df[df[hue_column] == hue_val]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=hue_data[col_j],
                                            y=hue_data[col_i],
                                            mode='markers',
                                            marker=dict(
                                                color=hue_color_map[hue_val],
                                                symbol=hue_symbol_map[hue_val],
                                                size=7,
                                                opacity=0.8,
                                                line=dict(width=0),  # No marker outline
                                            ),
                                            name=f"{hue_val}",
                                            showlegend=(i==0 and j==1),
                                            hoverinfo='none',  # Disable hover for cleaner look
                                        ),
                                        row=i+1, col=j+1
                                    )
                            else:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[col_j],
                                        y=df[col_i],
                                        mode='markers',
                                        marker=dict(
                                            color='rgba(76, 114, 176, 0.7)',  # Seaborn blue
                                            size=7,
                                            opacity=0.8,
                                            line=dict(width=0)  # No marker outline
                                        ),
                                        showlegend=False,
                                        hoverinfo='none'  # Disable hover for cleaner look
                                    ),
                                    row=i+1, col=j+1
                                )
                        
                        # Update axes labels
                        if i == n-1:  # Bottom row
                            fig.update_xaxes(title_text=col_j, row=i+1, col=j+1, 
                                           title_font=dict(size=12))
                        else:
                            fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)
                            
                        if j == 0:  # First column
                            fig.update_yaxes(title_text=col_i, row=i+1, col=j+1, 
                                           title_font=dict(size=12))
                        else:
                            fig.update_yaxes(showticklabels=False, row=i+1, col=j+1)
                
                # Aggiungi linee della griglia
                fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(211,211,211,0.5)')
                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(211,211,211,0.5)')
                
                # Update layout to be more like Seaborn's style
                fig.update_layout(
                    height=800,
                    width=800,
                    title="Pairplot with Transparent Density Plots on Diagonal",
                    plot_bgcolor='white',  # White background like Seaborn
                    paper_bgcolor='white',
                    margin=dict(l=40, r=20, t=40, b=40),
                    font=dict(family="Arial", size=12),
                    legend=dict(
                        title=dict(text=hue_column if hue_column != "None" else ""),
                        y=1.02,  # Posiziona la legenda sopra il grafico
                        x=1.02,  # Posiziona la legenda a destra del grafico
                        xanchor='left',
                        yanchor='bottom',
                        orientation='v',
                        bgcolor='rgba(255, 255, 255, 0.8)'
                    )
                )
                
                # Display the plot
                st.plotly_chart(fig)
                
                # Explanation
                st.markdown("""
                **Pairplot Interpretation**:
                * Each scatter plot shows the relationship between two variables
                * Diagonal plots show the transparent distribution of each variable
                * Look for patterns, clusters, and outliers in the relationships
                """)
            elif len(selected_columns) < 2 and st.session_state.get('button_clicked', False):
                st.warning("Please select at least 2 columns for the pairplot.")
        
        # Tab 2: Correlation Matrix
        with tab2:
            st.subheader("Correlation Matrix")
            st.markdown("Visualize the correlation between numerical variables")
            
            # Get numeric columns for correlation
            numeric_columns = get_numeric_columns(df)
            
            if len(numeric_columns) > 0:
                # Let the user select columns for the correlation matrix
                corr_columns = st.multiselect(
                    "Select numeric columns for correlation matrix",
                    numeric_columns,
                    default=numeric_columns[:8] if len(numeric_columns) > 8 else numeric_columns
                )
                
                # Correlation method selection
                corr_method = st.radio(
                    "Select correlation method",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True
                )
                
                if st.button("Generate Correlation Matrix") and len(corr_columns) >= 2:
                    # Create correlation matrix
                    correlation = df[corr_columns].corr(method=corr_method.lower())
                    
                    # Generate heatmap using plotly for better interactivity
                    fig = ff.create_annotated_heatmap(
                        z=correlation.values.round(2),
                        x=correlation.columns.tolist(),
                        y=correlation.index.tolist(),
                        annotation_text=correlation.values.round(2),
                        colorscale='Blues',
                        showscale=True
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{corr_method} Correlation Matrix",
                        height=600,
                        width=800
                    )
                    
                    # Display the interactive heatmap
                    st.plotly_chart(fig)
                    
                    # Explanation
                    st.markdown("""
                    **Correlation Matrix Interpretation**:
                    * Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation)
                    * 0 indicates no linear relationship
                    * Darker colors indicate stronger correlations
                    * **Pearson** measures linear relationships
                    * **Spearman** measures monotonic relationships (doesn't need to be linear)
                    * **Kendall** is another rank correlation that handles ties differently than Spearman
                    """)
                elif len(corr_columns) < 2 and st.session_state.get('corr_button_clicked', False):
                    st.warning("Please select at least 2 columns for the correlation matrix.")
            else:
                st.warning("No numeric columns found in the dataset for correlation analysis.")
        
        # Tab 3: Q-Q Plot
        with tab3:
            st.subheader("Q-Q Plot")
            st.markdown("Check if your data follows a normal distribution")
            
            # Get numeric columns for Q-Q plot
            numeric_cols = get_numeric_columns(df)
            
            if len(numeric_cols) > 0:
                # Select column for Q-Q plot
                qq_column = st.selectbox("Select a numeric column for Q-Q plot", numeric_cols)
                
                if st.button("Generate Q-Q Plot"):
                    # Create two columns for displaying plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # First, show histogram of the data
                        fig_hist = px.histogram(
                            df, x=qq_column, 
                            histnorm='probability density',
                            title=f"Histogram of {qq_column}",
                            labels={qq_column: qq_column}
                        )
                        # Add KDE curve
                        df_filtered = df[~df[qq_column].isna()]  # Remove NaN values
                        x_range = np.linspace(df_filtered[qq_column].min(), df_filtered[qq_column].max(), 1000)
                        kde = stats.gaussian_kde(df_filtered[qq_column])
                        y_range = kde(x_range)
                        fig_hist.add_scatter(x=x_range, y=y_range, mode='lines', name='KDE')
                        st.plotly_chart(fig_hist)
                    
                    with col2:
                        # Create Q-Q plot
                        fig_qq = plt.figure(figsize=(10, 6))
                        
                        # Remove NaN values for Q-Q plot
                        data = df[qq_column].dropna()
                        
                        # Generate the Q-Q plot
                        stats.probplot(data, dist="norm", plot=plt)
                        plt.title(f"Q-Q Plot for {qq_column}")
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig_qq)
                    
                    # Shapiro-Wilk test for normality
                    if len(data) <= 5000:  # Shapiro-Wilk has a sample size limitation
                        stat, p_value = stats.shapiro(data)
                        
                        st.markdown(f"""
                        **Shapiro-Wilk Test for Normality**:
                        * W-statistic: {stat:.4f}
                        * p-value: {p_value:.4f}
                        * Interpretation: The data {'does not appear to be' if p_value < 0.05 else 'may be'} normally distributed (at α=0.05)
                        """)
                    else:
                        # For larger datasets, use Anderson-Darling test
                        result = stats.anderson(data, dist='norm')
                        st.markdown(f"""
                        **Anderson-Darling Test for Normality**:
                        * A²-statistic: {result.statistic:.4f}
                        * Critical values: {result.critical_values}
                        * Significance levels: {result.significance_level}
                        """)
                    
                    # Explanation for Q-Q plot
                    st.markdown("""
                    **Q-Q Plot Interpretation**:
                    * Points following the diagonal line indicate the data follows a normal distribution
                    * Deviations from the line suggest departures from normality
                    * S-shaped patterns indicate skewness
                    * Curved patterns at the ends suggest heavy or light tails
                    """)
            else:
                st.warning("No numeric columns found in the dataset for Q-Q plot analysis.")
        
        # Tab 4: Parallel Coordinates Plot
        with tab4:
            st.subheader("Parallel Coordinates Plot")
            st.markdown("Visualize multidimensional numerical data and explore relationships between variables")
            
            # Get numeric columns for parallel coordinates
            num_cols = get_numeric_columns(df)
            
            if len(num_cols) > 0:
                # Let the user select columns for the parallel coordinates plot
                parallel_columns = st.multiselect(
                    "Select numeric columns for parallel coordinates plot",
                    num_cols,
                    default=num_cols[:5] if len(num_cols) > 5 else num_cols
                )
                
                # Color column selection
                color_options = ["None"] + all_columns
                color_column = st.selectbox(
                    "Select a column for color coding (categorical or numerical)",
                    color_options,
                    index=0
                )
                
                if st.button("Generate Parallel Coordinates Plot") and len(parallel_columns) >= 2:
                    # Create a copy of the dataframe with selected columns
                    plot_df = df[parallel_columns].copy()
                    
                    # Handle the color column
                    if color_column != "None":
                        # Check if color column is categorical
                        if not pd.api.types.is_numeric_dtype(df[color_column]):
                            # For categorical variables, create a numeric mapping
                            categories = df[color_column].unique()
                            cat_to_num = {cat: i for i, cat in enumerate(categories)}
                            color_values = df[color_column].map(cat_to_num)
                            
                            # Create a custom colorscale for categories
                            fig = px.parallel_coordinates(
                                plot_df,
                                dimensions=parallel_columns,
                                color=color_values,
                                color_continuous_scale=px.colors.qualitative.Plotly,
                                title="Parallel Coordinates Plot"
                            )
                            
                            # Add a legend for categorical colors
                            st.write("**Color Legend:**")
                            for cat, num in cat_to_num.items():
                                color_idx = num % len(px.colors.qualitative.Plotly)
                                color = px.colors.qualitative.Plotly[color_idx]
                                st.markdown(f"<span style='color:{color}'>■</span> {cat}", unsafe_allow_html=True)
                        else:
                            # For numeric color variables
                            fig = px.parallel_coordinates(
                                plot_df,
                                dimensions=parallel_columns,
                                color=df[color_column],
                                color_continuous_scale='Blues',
                                title="Parallel Coordinates Plot"
                            )
                    else:
                        # No color coding
                        fig = px.parallel_coordinates(
                            plot_df,
                            dimensions=parallel_columns,
                            title="Parallel Coordinates Plot"
                        )
                    
                    # Update layout
                    fig.update_layout(
                        height=600,
                        width=900
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig)
                    
                    # Explanation
                    st.markdown("""
                    **Parallel Coordinates Plot Interpretation**:
                    * Each vertical axis represents a different variable
                    * Each horizontal line represents an observation (row) in your dataset
                    * Lines that follow similar paths indicate similar observations
                    * Crossing lines between axes indicate negative correlations
                    * Parallel lines between axes indicate positive correlations
                    * Color coding helps identify patterns or clusters in your data
                    """)
                elif len(parallel_columns) < 2 and st.session_state.get('parallel_button_clicked', False):
                    st.warning("Please select at least 2 columns for the parallel coordinates plot.")
            else:
                st.warning("No numeric columns found in the dataset for parallel coordinates plot.")
                
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    # Show example images when no file is uploaded
    st.info("👆 Please upload an Excel file to start analyzing.")
    
    # Example images column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Pairplot Example")
        st.image("https://seaborn.pydata.org/_images/seaborn-pairplot-2.png", width=300)
    
    with col2:
        st.subheader("Correlation Matrix Example")
        st.image("https://miro.medium.com/max/1400/1*ucXgX8XwyIEa95rXYbhJBw.png", width=300)
    
    with col3:
        st.subheader("Q-Q Plot Example")
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Normal_normal_qq.svg", width=300)

    # Note about example images
    st.caption("Note: Example images are for illustration purposes and will be replaced by your data analysis.")