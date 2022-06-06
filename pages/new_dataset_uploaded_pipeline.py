import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pp import pp

def app():
    #st.header("New dataset uploaded pipeline")

    # chart_data = pd.DataFrame(
    #     [[1.10,0.80,0.68,0.60,0.50,0.40,0.40,0.45],[72,145,199,272,345,399,472,545]],
    #     columns=['a'])
    #
    # st.line_chart(chart_data)

    # #x = np.arange(500)

    dfs, column_names, columns_names2, columns_names2, path_list, time, mds = pp()

    #[72,145,199,272,345,399,472,545]
    #[1.10,0.80,0.68,0.60,0.50,0.40,0.40,0.45]
    source = pd.DataFrame({
        'Integration Length (samples)': dfs[4]['int_len'] ,
        'Perr': dfs[4]['prob_err']
    })

    c= alt.Chart(source).mark_line().encode(
        x='Integration Length (samples)',
        y='Perr'
    )

    st.subheader("Analysis noisetag_bci_201029_1340_ganglion dataset")
    st.altair_chart(c, use_container_width=True)

    st.write(pd.DataFrame({
        'StopErr': dfs[4]['se'],
        'StopThresh(P)': dfs[4]['st']
    }))