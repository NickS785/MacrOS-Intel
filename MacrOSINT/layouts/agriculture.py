#layouts/agriculture.py

# Fixed layout - ensure component IDs match callback expectations

from datetime import date
from MacrOSINT.data.data_tables import FASTable
from MacrOSINT.components import create_dd_menu
from MacrOSINT.components import FundamentalFrame as FFrame, FrameGrid
import pandas as pd
from typing import List
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px

def import_export_layout(data_type: str = "imports", data_source=None):
    """Generate layout for different data types (imports/exports)"""

    from MacrOSINT.data.data_tables import FASTable
    table_client = FASTable()

    # Get initial data and available commodities
    available_keys = table_client.available_keys()
    data_keys = [key for key in available_keys if key.endswith(f'{data_type}')]
    available_commodities = [key.split('/')[0] for key in data_keys]

    if not available_commodities:
        available_commodities = ['wheat', 'corn', 'rice']

    default_commodity = available_commodities[0]

    # Use provided data_source or get default data
    if data_source is not None:
        df = data_source
    else:
        df = table_client[f'{default_commodity}/{data_type}']

    # Extract unique countries and years for dropdowns
    countries = df.index.get_level_values('Partner').unique().tolist()
    years = df.index.get_level_values('date').year.unique().tolist()
    years.sort()

    return html.Div([
        # Header
        html.Div([
            html.H1(f"{data_type.title()} Sources Analysis Dashboard",
                    className='dashboard-title'),
            html.P(f"Analyze commodity {data_type} patterns by country and time period",
                   className='dashboard-subtitle'),

            # Commodity selector
            html.Div([
                html.Label("Select Commodity:", className='form-label'),
                dcc.Dropdown(
                    id='commodity-dropdown',  # ✓ Matches callback
                    options=[{'label': commodity.title(), 'value': commodity}
                             for commodity in available_commodities],
                    value=available_commodities[0],
                    className='commodity-dropdown dark-dropdown'
                ),
            ], className='commodity-selector-container'),

            # Store components
            dcc.Store(id='data-key-type', data={'key-type': data_type}),  # ✓ Matches callback
            dcc.Store(id='trade-data'),  # ✓ Matches callback

        ], className='dashboard-header'),

        # Main content container
        html.Div([
            # First row: Time series analysis
            html.Div([
                html.Div([
                    html.H3(f"Monthly {data_type.title()} Trends by Country",
                            className='section-title'),

                    # Controls for time series
                    html.Div([
                        html.Div([
                            html.Label("Select Country:", className='form-label'),
                            dcc.Dropdown(
                                id='country-dropdown',  # ✓ Matches callback
                                options=[{'label': country, 'value': country}
                                         for country in countries],
                                value=countries[0] if countries else None,
                                className='control-dropdown dark-dropdown'
                            )
                        ], className='control-columns_col-left'),

                        html.Div([
                            html.Label("Select Date Range:", className='form-label'),
                            dcc.DatePickerRange(
                                id='date-range-picker-trade',  # ✓ Fixed to match callback
                                start_date=date(2020, 1, 1),
                                end_date=date(2024, 12, 31),
                                display_format='YYYY-MM-DD',
                                className='date-picker dark-datepicker'
                            )
                        ], className='control-columns_col-right')
                    ], className='controls-row'),

                    # Time series chart
                    dcc.Graph(id='time-series-chart')  # ✓ Matches callback

                ], className='chart-card')
            ]),

            # Second row: Data sources breakdown
            html.Div([
                html.Div([
                    html.H3(f"{data_type.title()} Sources Breakdown by Year",
                            className='section-title'),

                    # Controls for breakdown
                    html.Div([
                        html.Div([
                            html.Label("Select Year:", className='form-label'),
                            dcc.Dropdown(
                                id='year-dropdown',  # ✓ Matches callback
                                options=[{'label': str(year), 'value': year}
                                         for year in years],
                                value=years[-1] if years else None,
                                className='control-dropdown dark-dropdown'
                            )
                        ], className='control-columns_col-left'),

                        html.Div([
                            html.Label("Chart Type:", className='form-label'),
                            dcc.RadioItems(
                                id='chart-type-radio',  # ✓ Fixed ID to match callback
                                options=[
                                    {'label': 'Pie Chart', 'value': 'pie'},
                                    {'label': 'Bar Chart', 'value': 'bar'}
                                ],
                                value='pie',
                                inline=True,
                                className='radio-group dark-radio'
                            )
                        ], className='control-columns_col-right')
                    ], className='controls-row'),

                    # Breakdown chart
                    dcc.Graph(id='breakdown-chart')  # ✓ Matches callback

                ], className='chart-card')
            ]),

            # Third row: Summary statistics
            html.Div([
                html.H3("Summary Statistics", className='section-title'),
                html.Div(id='summary-stats')  # ✓ Matches callback
            ], className='stats-card')

        ], className='main-content'),

    ], className='dashboard-container')


def psd_layout(commodity_key, FAS_table):
    comm_alias = FAS_table.esr_codes['alias'][commodity_key]
    import_export_cfg = [
        {
            'starting_key': f'{commodity_key}/psd/summary',
            'title': f'{comm_alias} Exports',
            'y_column': 'Exports',
            'chart_type': 'bar',
            'width': "45%",
            'height': '100%'
        },
        {'starting_key': f'{commodity_key}/psd/summary',
         'title': f'{comm_alias} Imports',
         'y_column': 'Imports',
         'chart_type': 'bar', 'width': "45%", 'height': '100%'}]
    beginning_ending_cfg = [
        {
            'starting_key': f'{commodity_key}/psd/summary',
            'y_column': 'Beginning Stocks',
            'chart_type': 'bar',
            'width': "45%",
            'height': "100%",
            'title': 'Total Stocks (start)'},
        {
            'starting_key': f'{commodity_key}/psd/summary',
            'y_column': 'Ending Stocks',
            'chart_type': 'bar',
            'width': "45%",
            'height': "100%",
            'title': 'Total Stocks (end)'
        }
    ]
    supply_demand = [
        {
            'starting_key': f'{commodity_key}/psd/summary',
            'y_column': 'Production',
            'title': "Domestic Production",
            'chart_type': 'bar',
            'width': "45%",
            'height': "100%",
        },
        {'starting_key': f'{commodity_key}/psd/summary',
         'y_column': 'Domestic Consumption',
         'title': "Domestic Consumption",
         'chart_type': 'bar',
         'width': "45%",
         'height': "100%", }
    ]
    trade_data = FFrame(FAS_table, import_export_cfg, layout='vertical', width="90%", div_prefix='intl-trade')
    domestic_data = FFrame(FAS_table, supply_demand, layout='vertical', width="90%", div_prefix='domestic-trade', )
    inventory_data = FFrame(FAS_table, beginning_ending_cfg, layout='horizontal', height="600px", div_prefix='inventory')
    dd_select_menu, comp_ids = create_dd_menu(FAS_table, header_text="Select Commodity")
    grid_cfg = {
        'layout_type': 'custom',
        'rows': 3,
        'cols': 2,
        'frame_positions': {
                0: {'row': 1, 'col': 1, 'col_span': 4},
                1: {'row': 1, 'col': 2, 'col_span': 4},
                2: {'row': 2, 'col': 1, 'col_span': 1}
                }
            }
    menu_config = {'enabled': False}
    Fgrid = FrameGrid(frames=[trade_data, domestic_data, inventory_data], grid_config=grid_cfg, menu_config=False)
    layout = [dd_select_menu, Fgrid.generate_layout()]
    menu_components = comp_ids
    return Fgrid, layout, menu_components

from MacrOSINT.components import FlexibleMenu

def create_esr_menu(table_client=None):
    """Create an ESR-specific menu."""

    # Initialize FASTable if no table_client provided
    if table_client is None:
        from MacrOSINT.data.data_tables import FASTable
        table_client = FASTable()

    menu = FlexibleMenu('esr_menu', position='right', width='350px', title='ESR Controls')

    # Commodity selector
    menu.add_dropdown('commodity', 'Commodity', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    # Year selector
    current_year = pd.Timestamp.now().year
    menu.add_dropdown('year', 'Marketing Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 2, current_year + 1)
    ], value=current_year)

    # Country filter
    menu.add_checklist('countries', 'Countries', [
        {'label': 'Korea, South', 'value': 'Korea, South'},
        {'label': 'Japan', 'value': 'Japan'},
        {'label': 'China', 'value': 'China'},
        {'label': 'Mexico', 'value': 'Mexico'},
        {'label': 'Canada', 'value': 'Canada'},
        {'label': 'Taiwan', 'value': 'Taiwan'}
    ], value=['Korea, South', 'Japan', 'China'])

    # Metric selector
    menu.add_dropdown('metric', 'Metric', [
        {'label': 'Weekly Exports', 'value': 'weeklyExports'},
        {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
        {'label': 'Gross New Sales', 'value': 'grossNewSales'},
        {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'}
    ], value='weeklyExports')

    # Apply button
    menu.add_button('apply', 'Apply Changes')

    return menu

def simple_esr_chart_update( chart_id: str, **menu_values):
    """
    Simple chart update function using Plotly Express for automatic country coloring.

    Args:
        chart_id: ID of the chart to update
        table_client: FASTable instance for data access
        **menu_values: Values from the menu (commodity, year, countries, metric, etc.)

    Returns:
        go.Figure: Updated figure
    """

    # Initialize FASTable if not provided


    table_client = FASTable()

    # Get menu values with defaults
    commodity = menu_values.get('commodity', 'cattle')
    year = menu_values.get('year', 2024)
    countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
    metric = menu_values.get('metric', 'weeklyExports')

    # Try to get real data from FASTable first
    try:
        # ESR data key format: {commodity}/exports/{year}
        esr_key = f"{commodity}/exports/{year}"
        data = table_client.get_key(esr_key)

        if data is not None and not data.empty:
            # Filter by selected countries
            if countries:
                data = data[data['country'].isin(countries)]

            # Ensure date columns_col is datetime
            if 'weekEndingDate' in data.columns:
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])

            # Filter to recent 26 weeks if too much data
            if len(data) > 200:
                latest_date = data['weekEndingDate'].max()
                cutoff_date = latest_date - pd.Timedelta(weeks=26)
                data = data[data['weekEndingDate'] >= cutoff_date]
        else:
            # Fall back to sample data if no real data available
            data = create_sample_data(commodity, year, countries, metric)

    except Exception as e:
        print(f"Error loading ESR data from FASTable: {e}")
        # Fall back to sample data
        data = create_sample_data(commodity, year, countries, metric)

    if data.empty:
        return create_empty_figure(f"{commodity.title()} - {metric} (MY {year})")

    # Use Plotly Express for automatic country coloring - SIMPLE!
    fig = px.line(
        data,
        x='weekEndingDate',
        y=metric,
        color='country',  # This automatically assigns different colors to each country
        title=f"{commodity.title()} - {metric.replace('_', ' ').title()} (MY {year})",
        markers=True
    )

    # Simple styling
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_sample_data(commodity: str, year: int, countries: List[str], metric: str) -> pd.DataFrame:
    """Create sample ESR data."""
    import numpy as np
    from datetime import datetime, timedelta

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=26)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')

    data_rows = []
    for country in countries:
        for date in dates:
            row = {
                'weekEndingDate': date,
                'country': country,
                'commodity': commodity,
                'weeklyExports': np.random.randint(1000, 8000),
                'outstandingSales': np.random.randint(20000, 50000),
                'grossNewSales': np.random.randint(500, 15000),
                'currentMYNetSales': np.random.randint(5000, 25000)
            }
            data_rows.append(row)

    return pd.DataFrame(data_rows)
def create_empty_figure(title: str) -> go.Figure:
    """Create empty figure with title."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(title=title, height=400)
    return fig
