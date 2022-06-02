import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.header("Performance tracking")

    # df = pd.DataFrame([[20.500, 15.513, 90.667], [57.833, 23.077, 34.667],[46.000, 32.436, 30.333],[36.667, 20.256, 25.222],[46.000, 28.718, 33.456]], columns=['kaggle', 'plos_one', 'lowlands'])
    # df.index.name = "Dataset-ID"
    # df.set_index([pd.Index([1, 2, 3, 4,5])])
    # data = df.reset_index().melt('Dataset-ID')
    # c = alt.Chart(data).mark_line().encode(
    #     x='Dataset-ID',
    #     y='value',
    #     color='variable'
    # )
    #test
    source = pd.DataFrame([[20.500, 15.513, 50.667], [57.833, 23.077, 60.667],[46.000, 55.436, 70.333],[36.667, 60.256, 80.222],[55.000, 85.718, 90.456]],
                          columns=['Kaggle', 'Plos_one', 'Lowlands'], index=pd.RangeIndex(5, name='Commit-ID'))
    source = source.reset_index().melt('Commit-ID', var_name='dataset', value_name='avg-AUDC')
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='Commit-ID:Q',
        y='avg-AUDC:Q',
        color='dataset:N'
    ).properties(
    title='Average AUDC across commits'
)
    st.altair_chart(line, use_container_width=True)
    # col1, col2 = st.columns(2)


    col1, col2 = st.columns([2, 1])
    col1.write(pd.DataFrame({'Commit-ID': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0],
        'SHA': ['0348784d9b0fbd9d595e31ae46d2e74632399507','8c0cdd85cd52268caf39d95cccced61e37705e56','df686161ce44ff255d154c42d34a91bcf4280ad8','b63df6a0b72e308e5f25fba24193ba00032028af','e8e4be5ba511ddaeac2a87fadc371bbef82cc270','20324aface79a6e431dc7d745284322fb2e94206'
            ,'fa836fb319a5589df5ea3c8e5918cba9830ff120','8d351b29946c3c1c5e450be2e573b8cc66795131','32786650c21a5f50cef61f303cdb50d99327c7a6','f4dd8c05dbbcb93c16a1e9c01d25d11d60d365fc','7cce378b0c90e70a3dc5aefd36c81956e126333a','72c5ac4acb19ad7559fb7200afc860093cc75cb0','a4a3f208771cf019073211bd0fc881e07d4f219a','01ab1895c0ff0407305ff6634cf4c5dbbd518fe7','01ab1895c0ff0407305ff6634cf4c5dbbd518fe7','01ab1895c0ff0407305ff6634cf4c5dbbd518fe7','19a897d985ff43f35f210facd889596c08d09a9a','20788484ad10d019110242e9713df4647b519603','20788484ad10d019110242e9713df4647b519603',
                '40a66722d28c51155095b9fff6add3bc373588b9','40a66722d28c51155095b9fff6add3bc373588b9']
    }))
    # col1.write(pd.DataFrame({'Data source': ['Kaggle', 'Plos_one', 'Lowlands'],
    #     'Dataset name': ['mindaffectBCI_noisetag_bci_201029_1340_ganglion.txt','Perr_plos_one_supervised_...','LL_eng_02_20170818_tr_train_1.mat']
    # }))

    source = pd.DataFrame({
        'Branch': ['open_source', 'master', 'wip'],
        '#commits': [20, 105, 63]
    })

    c= alt.Chart(source).mark_bar().encode(
        x='Branch',
        y='#commits'
    )
    col2.altair_chart(c, use_container_width=True)

    # col1, col2 = st.columns(2)
    # chart_data = pd.DataFrame(
    #     np.array([[1, 2, 3], [4, 5, 6],[4, 5, 6]]),
    #     columns=['open_source', 'master', 'wip'])
    #
    # col1.line_chart(chart_data)
    # #col2.chart_data.
    #
    # col2.write(pd.DataFrame({
    #     'first column': [1, 2, 3, 4],
    #     'second column': [10, 20, 30, 40]
    # }))
    #
    # col3, col4 = st.columns(2)
    # chart_data2 = pd.DataFrame(
    #     np.array([[20.500, 15.513, 90.667], [57.833, 23.077, 34.667],[46.000, 32.436, 30.333],[36.667, 20.256, 25.222],[46.000, 28.718, 33.456]]),
    #     columns=['kaggle', 'plos_one', 'lowlands'])
    #
    # # col3.line_chart(chart_data2)
    #
    # line_chart = alt.Chart(chart_data2).mark_line(interpolate='basis').encode(
    # alt.X('x', title='Year'),
    # alt.Y('y', title='Amount in liters'),
    # color='category:N'
    # ).properties(
    # title='Sales of consumer goods'
    # )
    #
    # st.altair_chart(line_chart)
