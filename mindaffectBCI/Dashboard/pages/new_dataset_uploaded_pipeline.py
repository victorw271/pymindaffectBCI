import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    #st.header("New dataset uploaded pipeline")

    # chart_data = pd.DataFrame(
    #     [[1.10,0.80,0.68,0.60,0.50,0.40,0.40,0.45],[72,145,199,272,345,399,472,545]],
    #     columns=['a'])
    #
    # st.line_chart(chart_data)

    # #x = np.arange(500)
    source = pd.DataFrame({
        'Integration Length (samples)': [72,145,199,272,345,399,472,545],
        'Perr': [1.10,0.80,0.68,0.60,0.50,0.40,0.40,0.45]
    })

    c= alt.Chart(source).mark_line().encode(
        x='Integration Length (samples)',
        y='Perr'
    )

    st.subheader("Analysis noisetag_bci_201029_1340_ganglion dataset")
    st.altair_chart(c, use_container_width=True)

    col1, col2 = st.columns([2,1])

    source = pd.DataFrame({
        'Metric': ['AUDC', 'PSAE', 'AUSC', 'SSAE'],
        'Value': [63, 34, 63, 31]
    })

    c= alt.Chart(source).mark_bar().encode(
        x='Metric',
        y='Value'
    )
    col1.altair_chart(c, use_container_width=True)


    col2.write(pd.DataFrame({
        'Metric': ['AUDC', 'PSAE', 'AUSC', 'SSAE'],
        'Value': [63.2, 34.1, 63.8, 31.7]
    }))

    st.write(pd.DataFrame({
        'StopErr': [1.00,  0.75,  0.68,  0.45,  0.47,  0.53,  0.45,  0.45],
        'StopThresh(P)': [0.95,  0.85,  0.82,  0.76,  0.73,  0.71,  0.65,  0.64 ]
    }))