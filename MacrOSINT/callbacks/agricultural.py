import pandas as pd
from dash import html
from plotly import graph_objects as go, express as px

from MacrOSINT.components.frames import FrameGrid
from MacrOSINT.utils.data_tools import store_to_df, df_to_store, get_sample_data, get_multi_year_sample_data, create_empty_figure
from MacrOSINT.data.data_tables import TableClient
from dash import Input, Output
import dash

class import_export_callbacks:

    def __init__(self, table_client: TableClient):
        self.table_client = table_client
        return

    def register_callbacks(self, app):
        table_client = self.table_client
        @app.callback(Output('commodity-dropdown', 'options'),
                      Output('commodity-dropdown', 'value'),
                      Input('data-key-type', 'data'))

        def update_dropdown(key_type):
            data_type = key_type['key-type']
            all_keys = table_client.available_keys()
            data_options = []
            available_keys = [key for key in all_keys if key.endswith(data_type)]
            for key in available_keys:
                data_options.append({'label':key.split('/')[0], 'value':key.split('/')[0]})
            return data_options, data_options[0]['value']

        @app.callback(
            [Output('trade-data', 'data'),
             Output('country-dropdown', 'options'),
             Output('country-dropdown', 'value'),
             Output('year-dropdown', 'options'),
             Output('year-dropdown', 'value')],
            [Input('commodity-dropdown', 'value'),
             Input('data-key-type', 'data')]
        )
        def update_commodity_data(selected_commodity, key_type):  # Removed 'self'
            # Get new data for the selected commodity using commodity/data_type pattern
            data_type = key_type['key-type']
            df_commodity = table_client[f'{selected_commodity}/{data_type}']

            # Extract unique countries and years for the new commodity
            countries_new = df_commodity.index.get_level_values('Partner').unique().tolist()
            years_new = df_commodity.index.get_level_values('date').year.unique().tolist()
            years_new.sort()

            # Prepare dropdown options
            country_options = [{'label': country, 'value': country} for country in countries_new]
            year_options = [{'label': str(year), 'value': year} for year in years_new]

            # Set default values
            default_country = countries_new[0] if countries_new else None
            default_year = years_new[-1] if years_new else None
            comm_data = df_to_store(df_commodity, reset_index=True, new_data={'commodity': selected_commodity})

            return comm_data, country_options, default_country, year_options, default_year

        @app.callback(
            Output('time-series-chart', 'figure'),
            [Input('country-dropdown', 'value'),
             Input('date-range-picker-trade', 'start_date'),  # Fixed ID
             Input('date-range-picker-trade', 'end_date'),  # Fixed ID
             Input('trade-data', 'data'),
             Input('data-key-type', 'data')],
            prevent_initial_call=True
        )
        def update_time_series(selected_country, start_date, end_date, trade_data, key_type):  # Removed 'self'
            # Define colors
            colors = {
                'primary': '#64B5F6',
                'secondary': '#E57373',
                'background': '#121212',
                'card_background': '#1E1E1E',
                'text': '#FFFFFF',
                'text_secondary': '#B0B0B0',
                'accent': '#FFB74D',
                'grid': '#2A2A2A',
                'border': '#333333'
            }

            # Handle None or empty trade_data
            if not trade_data:
                fig = go.Figure()
                fig.update_layout(
                    title="No data available",
                    plot_bgcolor=colors['card_background'],
                    paper_bgcolor=colors['card_background'],
                    font=dict(color=colors['text'])
                )
                return fig

            # Get commodity data using commodity/data_type pattern
            df_commodity = store_to_df(trade_data, index_column=['Partner', 'date'], datetime_date_col=True)
            selected_commodity = df_commodity['commodity'].iloc[0]  # Changed from -1 to 0

            # Filter data for selected country and date range
            if selected_country not in df_commodity.index.get_level_values('Partner'):
                # Return empty figure if country not available for this commodity
                fig = go.Figure()
                fig.update_layout(
                    title=f"No data available for {selected_country} in {selected_commodity.title()}",
                    plot_bgcolor=colors['card_background'],
                    paper_bgcolor=colors['card_background'],
                    font=dict(color=colors['text'])
                )
                return fig

            country_data = df_commodity.loc[selected_country].copy()

            # Convert date strings to datetime if needed
            if start_date and end_date:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                country_data = country_data[(country_data.index >= start_date) &
                                            (country_data.index <= end_date)]

            # Create time series plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=country_data.index,
                y=country_data['value'],
                mode='lines+markers',
                name=f"{selected_country} ({selected_commodity.title()})",
                line=dict(color=colors['primary'], width=3),
                marker=dict(size=6, color=colors['primary'])
            ))

            fig.update_layout(
                title=f"Monthly {selected_commodity.title()} {key_type['key-type'].title()} Values - {selected_country}",
                xaxis_title="Date",
                yaxis_title=f"{key_type['key-type'].title()} Value",
                hovermode='x unified',
                plot_bgcolor=colors['card_background'],
                paper_bgcolor=colors['card_background'],
                font=dict(color=colors['text']),
                showlegend=False,
                title_font=dict(color=colors['text'], size=16)
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'],
                             color=colors['text'], tickcolor=colors['text'])
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'],
                             color=colors['text'], tickcolor=colors['text'])

            return fig

        @app.callback(
            Output('breakdown-chart', 'figure'),
            [Input('trade-data', 'data'),
             Input('year-dropdown', 'value'),
             Input('chart-type-radio', 'value'),  # Fixed ID (removed key_type-)
             Input('data-key-type', 'data')],
            prevent_initial_call=True
        )
        def update_breakdown(trade_data, selected_year, chart_type, key_type):  # Removed 'self'
            # Define colors
            colors = {
                'primary': '#64B5F6',
                'secondary': '#E57373',
                'background': '#121212',
                'card_background': '#1E1E1E',
                'text': '#FFFFFF',
                'text_secondary': '#B0B0B0',
                'accent': '#FFB74D',
                'grid': '#2A2A2A',
                'border': '#333333'
            }

            # Handle None or empty trade_data
            if not trade_data:
                fig = go.Figure()
                fig.update_layout(
                    title="No data available",
                    plot_bgcolor=colors['card_background'],
                    paper_bgcolor=colors['card_background'],
                    font=dict(color=colors['text'])
                )
                return fig

            # Get commodity data using commodity/data_type pattern
            df_commodity = store_to_df(trade_data, index_column=['Partner', 'date'],
                                       datetime_date_col=True)  # Changed from -1 to 0
            selected_commodity = df_commodity['commodity'].iloc[0]
            data_type = key_type['key-type']

            # Filter data for selected year
            year_data = df_commodity[df_commodity.index.get_level_values('date').year == selected_year].copy()

            if year_data.empty:
                # Return empty figure if no data for this year
                fig = go.Figure()
                fig.update_layout(
                    title=f"No data available for {selected_year} in {selected_commodity.title()}",
                    plot_bgcolor=colors['card_background'],
                    paper_bgcolor=colors['card_background'],
                    font=dict(color=colors['text'])
                )
                return fig

            # Group by country and sum values
            country_totals = year_data.groupby('Partner')['value'].sum().sort_values(ascending=False)

            if chart_type == 'pie':
                fig = px.pie(
                    values=country_totals.values,
                    names=country_totals.index,
                    title=f"{selected_commodity.title()} {data_type.title()} Sources Distribution - {selected_year}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label',
                                  textfont=dict(color='white', size=12))

            else:  # bar chart
                fig = px.bar(
                    x=country_totals.index,
                    y=country_totals.values,
                    title=f"{selected_commodity.title()} {data_type.title()} Values by Country - {selected_year}",
                    color=country_totals.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title=f"Total {data_type.title()} Value",
                    showlegend=False
                )
                fig.update_xaxes(tickangle=45, color=colors['text'], tickcolor=colors['text'])
                fig.update_yaxes(color=colors['text'], tickcolor=colors['text'])

            fig.update_layout(
                plot_bgcolor=colors['card_background'],
                paper_bgcolor=colors['card_background'],
                font=dict(color=colors['text']),
                title_font=dict(color=colors['text'], size=16)
            )

            return fig

        @app.callback(
            Output('summary-stats', 'children'),
            [Input('country-dropdown', 'value'),
             Input('year-dropdown', 'value'),
             Input('trade-data', 'data'),
             Input('data-key-type', 'data')],
            prevent_initial_call=True  # Fixed typo: was 'prevent_inital_call'
        )
        def update_summary_stats(selected_country, selected_year, trade_data, key_type):  # Removed 'self'
            # Define colors
            colors = {
                'primary': '#64B5F6',
                'secondary': '#E57373',
                'background': '#121212',
                'card_background': '#1E1E1E',
                'text': '#FFFFFF',
                'text_secondary': '#B0B0B0',
                'accent': '#FFB74D',
                'grid': '#2A2A2A',
                'border': '#333333'
            }

            # Handle None or empty trade_data
            if not trade_data:
                return html.Div("No data available", style={'color': colors['text']})

            # Get commodity data using commodity/data_type pattern
            df_commodity = store_to_df(trade_data, index_column=['Partner', 'date'],
                                       datetime_date_col=True)  # Changed from -1 to 0
            selected_commodity = df_commodity['commodity'].iloc[0]  # Changed from -1 to 0
            data_type = key_type['key-type']

            # Overall statistics for the commodity
            total_value = df_commodity['value'].sum()
            avg_monthly = df_commodity['value'].mean()

            # Country-specific statistics
            country_total = 0
            country_avg = 0
            country_share = 0
            if selected_country and selected_country in df_commodity.index.get_level_values('Partner'):
                country_data = df_commodity.loc[selected_country]
                country_total = country_data['value'].sum()
                country_avg = country_data['value'].mean()
                country_share = (country_total / total_value) * 100 if total_value > 0 else 0

            # Year-specific statistics
            year_data = df_commodity[df_commodity.index.get_level_values('date').year == selected_year]
            year_total = year_data['value'].sum()
            top_country = "N/A"
            num_sources = 0
            if not year_data.empty:
                top_country = year_data.groupby('Partner')['value'].sum().idxmax()
                num_sources = len(year_data.index.get_level_values('Partner').unique())

            return html.Div([
                html.Div([
                    html.Div([
                        html.H4(f"{selected_commodity.title()} - Overall Statistics",
                                style={'color': colors['primary']}),
                        html.P(f"Total {data_type.title()}: ${total_value:,.0f}", style={'color': colors['text']}),
                        html.P(f"Average Monthly: ${avg_monthly:,.0f}", style={'color': colors['text']})
                    ], className="four columns"),

                    html.Div([
                        html.H4(f"{selected_country} Statistics", style={'color': colors['secondary']}),
                        html.P(f"Total {data_type.title()}: ${country_total:,.0f}", style={'color': colors['text']}),
                        html.P(f"Average Monthly: ${country_avg:,.0f}", style={'color': colors['text']}),
                        html.P(f"Share of Total: {country_share:.1f}%", style={'color': colors['text']})
                    ], className="four columns"),

                    html.Div([
                        html.H4(f"{selected_year} Statistics", style={'color': colors['accent']}),
                        html.P(f"Year Total: ${year_total:,.0f}", style={'color': colors['text']}),
                        html.P(f"Top Source: {top_country}", style={'color': colors['text']}),
                        html.P(f"Number of Sources: {num_sources}", style={'color': colors['text']})
                    ], className="four columns")
                ], className="row")
            ])

def register_psd_callbacks(Fgrid:FrameGrid, component_ids:dict, FAS_table):
    app = dash.get_app()
    output_charts = []
    for frame in Fgrid.frames:
        output_charts.extend(frame.charts)

    @app.callback([Output(chart.chart_id, 'figure') for chart in output_charts],
                  Input(component_ids['dd'], 'value'),
                  Input(component_ids['load-btn'], 'n_clicks')
                  )
    def get_psd_figures(key, n_clicks):
        commodity = key.split('/')[0]
        if n_clicks:
            df = FAS_table[key]
            update_columns = ['Exports', 'Imports', 'Production', 'Domestic Consumption', 'Beginning Stocks',
                              'Ending Stocks']
            charts = []
            for frame in Fgrid.frames:
                charts.extend(frame.charts)
            new_figures = []
            for n in range(len(charts)):
                if hasattr(charts[n], 'title'):
                    charts[
                        n].title = f'{FAS_table.esr_codes["alias"][commodity.lower()].capitalize()} {update_columns[n]}'

                new_figures.append(
                    charts[n].update_data_source(df, y_column=update_columns[n])
                )

            return new_figures

def create_esr_chart_update_functions(app_instance):
    """
    Create chart update functions with access to the app's data storage.

    Args:
        app_instance: The ESRAnalysisApp instance

    Returns:
        dict: Dictionary of update functions for each page
    """
    def sales_trends_chart_update(chart_id: str, **menu_values):
        """Update function for Sales Trends page - each chart gets its specific metric."""
        commodity = menu_values.get('commodity', 'cattle')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])

        # Get data
        current_year = pd.Timestamp.now().year
        data = get_sample_data(commodity, current_year, countries)

        # Filter by selected countries
        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"{commodity.title()} - Sales Trends")

        # EXPLICIT CHART-TO-METRIC MAPPING
        if chart_id == 'sales_trends_chart_0':
            metric = 'weeklyExports'
            metric_name = 'Weekly Exports'
        elif chart_id == 'sales_trends_chart_1':
            metric = 'outstandingSales'
            metric_name = 'Outstanding Sales'
        elif chart_id == 'sales_trends_chart_2':
            metric = 'grossNewSales'
            metric_name = 'Gross New Sales'
        else:
            metric = 'weeklyExports'  # fallback
            metric_name = 'Weekly Exports'

        # Create chart with the SPECIFIC metric
        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,  # This is the key - use the specific metric
            color='country',
            title=f"{commodity.title()} - {metric_name}",
            markers=True
        )

        fig.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def country_analysis_chart_update(chart_id: str, **menu_values):
        """Update function for Country Analysis page - multi-year, single country."""
        commodity = menu_values.get('commodity', 'cattle')
        country = menu_values.get('country', 'Korea, South')

        # Get multi-year data
        data = get_multi_year_sample_data(commodity, country)

        # Filter by selected country
        data = data[data['country'] == country]

        if data.empty:
            return create_empty_figure(f"{commodity.title()} - {country} Analysis")

        # EXPLICIT CHART-TO-METRIC MAPPING
        if chart_id == 'country_analysis_chart_0':
            metric = 'weeklyExports'
            metric_name = 'Weekly Exports'
        elif chart_id == 'country_analysis_chart_1':
            metric = 'outstandingSales'
            metric_name = 'Outstanding Sales'
        else:
            metric = 'weeklyExports'  # fallback
            metric_name = 'Weekly Exports'

        # Create chart colored by marketing year
        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,  # Use the specific metric
            color='marketing_year',
            title=f"{commodity.title()} - {country} {metric_name} (5-Year)",
            markers=True
        )

        fig.update_layout(
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def commitment_analysis_chart_update(chart_id: str, **menu_values):
        """Update function for Commitment Analysis page - each chart gets its specific commitment metric and chart type."""
        commodity = menu_values.get('commodity', 'cattle')
        year = menu_values.get('year', pd.Timestamp.now().year)
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])

        # Get data
        data = get_sample_data(commodity, year, countries)

        # Filter by selected countries
        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"{commodity.title()} - Commitment Analysis (MY {year})")

        # EXPLICIT CHART-TO-METRIC AND CHART-TYPE MAPPING FOR COMMITMENT METRICS
        if chart_id == 'commitment_analysis_chart_0':
            metric = 'currentMYTotalCommitment'
            metric_name = 'Current MY Total Commitment'
            chart_type = 'area'
        elif chart_id == 'commitment_analysis_chart_1':
            metric = 'currentMYNetSales'
            metric_name = 'Current MY Net Sales'
            chart_type = 'line'
        elif chart_id == 'commitment_analysis_chart_2':
            metric = 'nextMYOutstandingSales'
            metric_name = 'Next MY Outstanding Sales'
            chart_type = 'bar'
        elif chart_id == 'commitment_analysis_chart_3':
            metric = 'nextMYNetSales'
            metric_name = 'Next MY Net Sales'
            chart_type = 'line'
        else:
            metric = 'currentMYTotalCommitment'  # fallback
            metric_name = 'Current MY Total Commitment'
            chart_type = 'area'

        # Create chart based on the specified chart type
        if chart_type == 'area':
            fig = px.area(
                data,
                x='weekEndingDate',
                y=metric,
                color='country',
                title=f"{commodity.title()} - {metric_name} (MY {year})"
            )
        elif chart_type == 'bar':
            fig = px.bar(
                data,
                x='weekEndingDate',
                y=metric,
                color='country',
                title=f"{commodity.title()} - {metric_name} (MY {year})"
            )
        else:  # line chart
            fig = px.line(
                data,
                x='weekEndingDate',
                y=metric,
                color='country',
                title=f"{commodity.title()} - {metric_name} (MY {year})",
                markers=True
            )

        fig.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def comparative_analysis_chart_update(chart_id: str, **menu_values):
        """Update function for Comparative Analysis page - different commodities, user-selected metric."""
        commodity_a = menu_values.get('commodity_a', 'cattle')
        commodity_b = menu_values.get('commodity_b', 'corn')
        year = menu_values.get('year', pd.Timestamp.now().year)
        metric = menu_values.get('metric', 'weeklyExports')  # User selects this
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])

        # Determine which commodity based on frame
        if 'comparison_frame1' in chart_id:
            commodity = commodity_a
            frame_label = "A"
        else:
            commodity = commodity_b
            frame_label = "B"

        # Get data for the specific commodity
        data = get_sample_data(commodity, year, countries)

        # Filter by selected countries
        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"Commodity {frame_label}: {commodity.title()}")

        # Use the metric selected by the user in the menu
        metric_name = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales'
        }.get(metric, metric.replace('_', ' ').title())

        # Create chart with countries colored automatically
        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,  # Use the user-selected metric
            color='country',
            title=f"Commodity {frame_label}: {commodity.title()} - {metric_name} (MY {year})",
            markers=True
        )

        fig.update_layout(
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    return {
        'sales_trends': sales_trends_chart_update,
        'country_analysis': country_analysis_chart_update,
        'commitment_analysis': commitment_analysis_chart_update,
        'comparative_analysis': comparative_analysis_chart_update
    }


