import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt

def simulate_roi(n, alpha, beta, theta, lambda_, clv):
    conversion_rates = np.random.beta(alpha, beta, n)
    cpcs = np.random.gamma(theta, 1/lambda_, n)
    conversions = np.random.binomial(1, conversion_rates)
    roi = np.where(conversions, clv - cpcs, -cpcs)
    return roi

st.set_page_config(layout="wide")

# Beautiful header
st.markdown(
    """
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #f63366;  /* Use the slider color */
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: normal;
        color: #555555;
        text-align: center;
        margin-bottom: 40px;
    }
    .explanation {
        font-size: 16px;
        color: white;
        margin-bottom: 20px;
    }
    .contact {
        font-size: 16px;
        color: #888888;
        text-align: center;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">Campaign Performance Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Simulate and visualize the performance of your marketing campaigns</div>', unsafe_allow_html=True)

# Method explanation
st.markdown(
    """
    <div class="explanation">
    A website visitor either converts with a certain probability or doesn't. If the user converts, we get the customer lifetime value (CLV) minus the cost per click (CPC). If the user doesn't convert, we just pay the CPC. We can calculate the revenue of a campaign over time by adding up these numbers for all web traffic funneled by the campaign.
    <br><br>
    This uses some simplifying assumptions which are not true in the actual world:
    <ul>
        <li>CPC and conversion probability of a visitor are independent.</li>
        <li>CPC of all visitors are independent of each other which can cause diminishing returns </li>
        <li>Does not estimate incremental revenue.</li>
        <li>Multiple platforms or campaigns might assist a conversion.</li>
        <li>Does not account for seasonality or trends.</li>
        <li>....</li>
    </ul>
    So only use this as a starting point but let me know if you have any suggestions or improvements!
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for parameters
st.sidebar.header('Simulation Parameters')
alpha = st.sidebar.slider('Alpha (Beta Distribution)', 1, 100, 3)
beta = st.sidebar.slider('Beta (Beta Distribution)', 1, 100, 97)
theta = st.sidebar.slider('Theta (Gamma Distribution)', 1, 20, 10)
lambda_ = st.sidebar.slider('Lambda (Gamma Distribution)', 0.1, 1.0, 0.5)
clv = st.sidebar.number_input('Customer Lifetime Value', value=60)
N = st.sidebar.number_input('Number of Visits', value=100)
k = st.sidebar.number_input('Number of simulations', value=150)

# Distribution plots
st.header('Parameter Distributions')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Conversion Rate Distribution (Beta)')
    x_beta = np.linspace(0, 1, 1000)
    y_beta = stats.beta.pdf(x_beta, alpha, beta)
    fig_beta = go.Figure(go.Scatter(x=x_beta, y=y_beta, mode='lines', fill='tozeroy'))
    fig_beta.update_layout(title='Beta Distribution', xaxis_title='Conversion Rate', yaxis_title='Probability Density')
    st.plotly_chart(fig_beta, use_container_width=True)
    beta_mean = alpha / (alpha + beta)
    st.markdown(f"**Average Conversion Rate: {beta_mean * 100:.2f}%**")

with col2:
    st.subheader('CPC Distribution (Gamma)')
    # Calculate the range for x-axis to cover at least 99% of the area
    x_gamma_max = stats.gamma.ppf(0.999, a=theta, scale=1/lambda_)
    x_gamma = np.linspace(0, x_gamma_max, 1000)
    y_gamma = stats.gamma.pdf(x_gamma, a=theta, scale=1/lambda_)
    fig_gamma = go.Figure(go.Scatter(x=x_gamma, y=y_gamma, mode='lines', fill='tozeroy'))
    fig_gamma.update_layout(title='Gamma Distribution', xaxis_title='Cost per Click', yaxis_title='Probability Density')
    st.plotly_chart(fig_gamma, use_container_width=True)
    gamma_mean = theta / lambda_
    st.markdown(f"**Average CPC Value: {gamma_mean:.4f}**")

# Simulation Results
st.header('Simulation Results')

# Button to run simulation
if st.button('Run Simulation'):
    # Run simulation
    roi_matrix = np.array([simulate_roi(N, alpha, beta, theta, lambda_, clv) for _ in range(k)])
    cumulative_roi = np.cumsum(roi_matrix, axis=1)

    # Plotting
    fig = go.Figure()
    
    # Get a colormap
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / k) for i in range(k)]

    # Plot each time series with different colors
    for i in range(k):
        color = f'rgba({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)}, 0.5)'
        fig.add_trace(go.Scatter(y=cumulative_roi[i], mode='lines', 
                                 line=dict(width=1, color=color), 
                                 showlegend=False))

    # Plot mean cumulative ROI
    mean_cumulative_roi = np.mean(cumulative_roi, axis=0)
    fig.add_trace(go.Scatter(y=mean_cumulative_roi, mode='lines', name='Mean Revenue Time Series',
                             line=dict(color='red', width=2)))

    fig.update_layout(
        title=f'Revenue for {k} Simulations over {N} Website Visits',
        xaxis_title='Number of Website Visits',
        yaxis_title='Revenue',
        showlegend=True,
        height=600,
        plot_bgcolor='black',  # Set plot background to black
        paper_bgcolor='black',  # Set paper background to black
        font=dict(color='white'),  # Set font color to white for better contrast
        yaxis=dict(showgrid=True, zeroline=True, showticklabels=True),  # Restore y-axis elements
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Click 'Run Simulation' to see results.")

# Contact information
st.markdown('<div class="contact">For suggestions, features, or bugs, contact me at mail.hashiromer.com</div>', unsafe_allow_html=True)
