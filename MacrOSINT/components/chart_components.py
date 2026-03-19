from typing import Optional, Tuple, Dict
from MacrOSINT.data.data_tables import MarketTable
from dash import dcc, html
from datetime import  datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class MarketChart:
    """Enhanced chart component with timeframe resampling and indicator support."""

    def __init__(self, chart_id='market-chart', title="Market Chart", chart_type='line', height=300,
                 line_color='#1f77b4', resample_freq=None, start_date=None, end_date=None, config=None):
        if not config:
            config = {}

        self.chart_id = chart_id
        self.title = config.get('title', title)
        self.chart_type = config.get('chart_type', 'line')
        self.height = config.get('height', height)
        self.line_color = config.get('line_color', line_color)
        self.indicators = []
        self.market_table = MarketTable()
        self.ticker = config.get('ticker', None)
        self.interval = config.get('freq', resample_freq)
        self.resample = True if self.interval else False
        self.start_date = config.get('start_date', start_date)
        self.end_date = config.get('end_date', end_date)
        self.ohlc_agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        self.original_data = None
        self.market_data = None

    def _apply_date_filter(self, data):
        """Apply date range filtering."""
        if data is None or data.empty or (not self.start_date and not self.end_date):
            return data

        filtered_data = data.copy()

        if isinstance(filtered_data.index, pd.DatetimeIndex):
            if self.start_date:
                filtered_data = filtered_data[filtered_data.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                filtered_data = filtered_data[filtered_data.index <= pd.to_datetime(self.end_date)]
        elif 'Date' in filtered_data.columns:
            filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
            if self.start_date:
                filtered_data = filtered_data[filtered_data['Date'] >= pd.to_datetime(self.start_date)]
            if self.end_date:
                filtered_data = filtered_data[filtered_data['Date'] <= pd.to_datetime(self.end_date)]

        return filtered_data

    def _apply_resampling(self, data):
        """Apply timeframe resampling with OHLC aggregation."""
        if data is None or data.empty or not self.interval:
            return data

        resampled_data = data.copy()

        if not isinstance(resampled_data.index, pd.DatetimeIndex):
            if 'Date' in resampled_data.columns:
                resampled_data = resampled_data.set_index(pd.to_datetime(resampled_data['Date']))
            else:
                return data

        # Build aggregation dictionary
        agg_dict = {}
        for col in resampled_data.columns:
            if col in self.ohlc_agg:
                agg_dict[col] = self.ohlc_agg[col]
            elif resampled_data[col].data_type in ['float64', 'int64', 'float32', 'int32']:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'last'

        try:
            resampled_data = resampled_data.resample(self.interval).agg(agg_dict)
            resampled_data = resampled_data.dropna(how='all').reset_index()
            return resampled_data
        except Exception as e:
            print(f"Resampling error: {e}")
            return data

    def _prepare_chart_data(self):
        """Apply filtering and resampling pipeline."""
        if self.original_data is None:
            return self.market_data

        chart_data = self.original_data.copy()
        chart_data = self._apply_date_filter(chart_data)
        chart_data = self._apply_resampling(chart_data)
        return chart_data

    def _get_x_data(self, data):
        """Extract x-axis data from dataframe."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index
        elif 'date' in data.columns:
            return data['date']
        elif 'Date' in data.columns:
            return data['Date']
        else:
            return np.arange(0,len(data))

    def _get_title_with_info(self):
        """Generate title with timeframe and date range info."""
        title = self.title
        if self.interval:
            title += f" ({self.interval})"
        if self.start_date or self.end_date:
            start = self.start_date or 'Start'
            end = self.end_date or 'End'
            title += f" [{start} to {end}]"
        return title

    def get_chart_figure(self):
        """Generate the chart figure."""
        chart_data = self._prepare_chart_data()

        if chart_data is None or chart_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig

        x_data = self._get_x_data(chart_data)

        # Handle indicators with subplots
        separate_indicators = [ind for ind in self.indicators if ind.get('axis') == 'separate']
        if separate_indicators:
            return self._create_subplot_chart(chart_data, x_data)

        # Create main chart
        fig = go.Figure()

        if self.chart_type == 'candlestick' and all(
                col in chart_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(go.Candlestick(x=x_data, open=chart_data['Open'], high=chart_data['High'],
                                         low=chart_data['Low'], close=chart_data['Close'], name='OHLC'))
        else:
            # Line chart
            y_col = 'Close' if 'Close' in chart_data.columns else ('Price' if 'Price' in chart_data.columns else
                                                                   chart_data.select_dtypes(
                                                                       include=[np.number]).columns[0] if len(
                                                                       chart_data.select_dtypes(
                                                                           include=[np.number]).columns) > 0 else None)

            if y_col:
                fig.add_trace(go.Scatter(x=x_data, y=chart_data[y_col], mode='lines', name=y_col,
                                         line=dict(color=self.line_color, width=2)))

        # Add overlay indicators
        overlay_indicators = [ind for ind in self.indicators if ind.get('axis') != 'separate']
        for indicator in overlay_indicators:
            self._add_indicator_traces(fig, indicator)

        fig.update_layout(
            title=self._get_title_with_info(),
            xaxis_title="Date", yaxis_title="Price ($)",
            height=self.height, margin=dict(l=40, r=40, t=40, b=40),
            xaxis_rangeslider_visible=(self.chart_type == 'line')
        )

        return fig

    def _create_subplot_chart(self, chart_data, x_data):
        """Create chart with subplots for separate indicators."""
        separate_indicators = [ind for ind in self.indicators if ind.get('axis') == 'separate']
        num_subplots = 1 + len(separate_indicators)

        row_heights = [0.7] + [0.3 / len(separate_indicators)] * len(separate_indicators)
        subplot_titles = [self.title] + [ind.get('name', f'Indicator {i + 1}') for i, ind in
                                         enumerate(separate_indicators)]

        fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=subplot_titles, row_heights=row_heights)

        # Add main chart
        if self.chart_type == 'candlestick' and all(
                col in chart_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(go.Candlestick(x=x_data, open=chart_data['Open'], high=chart_data['High'],
                                         low=chart_data['Low'], close=chart_data['Close'], name='OHLC'), row=1, col=1)
        else:
            y_col = 'Close' if 'Close' in chart_data.columns else chart_data.select_dtypes(include=[np.number]).columns[
                0]
            fig.add_trace(go.Scatter(x=x_data, y=chart_data[y_col], mode='lines', name=y_col,
                                     line=dict(color=self.line_color, width=2)), row=1, col=1)

        # Add overlay indicators to main chart
        overlay_indicators = [ind for ind in self.indicators if ind.get('axis') != 'separate']
        for indicator in overlay_indicators:
            self._add_indicator_traces(fig, indicator, row=1, col=1)

        # Add separate indicators
        for i, indicator in enumerate(separate_indicators):
            self._add_indicator_traces(fig, indicator, row=i + 2, col=1)

        fig.update_layout(title=self._get_title_with_info(), height=self.height,
                          margin=dict(l=40, r=40, t=40, b=40), xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)

        return fig

    def _add_indicator_traces(self, fig, indicator, row=None, col=None):
        """Add indicator traces to figure."""
        data = indicator.get('data')
        y_columns = indicator.get('y_columns', [])
        colors = indicator.get('colors', ['#ff7f0e', '#2ca02c', '#d62728'])

        if data is None or data.empty or not y_columns:
            return

        x_data = self._get_x_data(data)

        for i, column in enumerate(y_columns):
            if column not in data.columns:
                continue

            trace = go.Scatter(x=x_data, y=data[column], mode='lines', name=column,
                               line=dict(color=colors[i % len(colors)], width=1.5))

            if row and col:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

    def plot_indicator(self, config):
        """Add indicator to chart."""
        required_keys = ['key_type', 'axis', 'data', 'y_columns']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in indicator config")

        config.setdefault('name', f"Indicator {len(self.indicators) + 1}")
        self.indicators.append(config)
        return self.get_chart_figure()

    def set_timeframe_config(self, resample_freq=None, start_date=None, end_date=None):
        """Update timeframe configuration."""
        self.interval = resample_freq
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = self._prepare_chart_data()
        return self.get_chart_figure()

    def load_ticker_data(self):
        """Load market data for ticker."""
        if self.market_table:
            data = self.market_table.get_historical(self.ticker, self.start_date, self.end_date, self.resample, self.interval)
            if data is not None:
                try:
                    data['Date'] = data.index
                except Exception as e:
                    print(f'Exception No date index')
                self.original_data = data.copy()
                self.market_data = self._prepare_chart_data()
                return self.get_chart_figure()
        return None

class FundamentalChart:
    """
    A reusable supply and demand chart component that reads data from HDF5 files.

    Parameters:
    - chart_id: Unique identifier for the chart
    - title: Chart title
    - hdf_file_path: Path to the HDF5 file
    - hdf_key: Key to access the dataframe in the HDF5 file
    - y_column: Column name for y-axis data (default: 'Value')
    - x_column: Column name for x-axis data (default: 'Date')
    - chart_type: 'bar', 'line', or 'area'
    - width: Width as percentage string (e.g., '49%') or pixels (e.g., '400px')
    - height: Height in pixels (default: 300)
    - float_position: 'left', 'right', or None for CSS float positioning
    - margin: Margin string (e.g., '5px 1%')
    """

    def __init__(self, chart_id, data=None, config=None, title="Supply/Demand Chart",
                 x_column='Date', chart_type='bar',
                 width='49%', line_color='blue', height=300, float_position='left', margin='5px 1%',
                 theme='plotly_dark'):
        config = {} if config is None else config
        self.config = config
        self.chart_id = chart_id
        self.line_color = config.get('line_color', line_color)
        self.title = config.get('title', title)
        self.data = config.get('data', None)
        self.chart_type = config.get('chart_type', chart_type)
        self.width = config.get('width', width)
        self.height = config.get('height', height)
        self.float_position = config.get('float_position', float_position)
        self.margin = config.get('margin', margin)
        self.y_column = config.get('y_column', None)
        self.x_column = config.get('x_column', x_column)
        self.theme = config.get('theme', theme)

    def _load_data(self):
        """Load data from HDF5 file using TableClient"""
        return

    def _create_bar_chart(self):
        """Create a bar chart from the data"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

                fig.add_trace(go.Bar(
                    x=x_data,
                    y=self.data[self.y_column],
                    name=self.y_column,
                    marker_color=f'{self.line_color}'
                ))
            else:
                # Show available columns if y_column not found
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            # Placeholder data
            fig.add_trace(go.Bar(x=[], y=[], name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )

        return fig

    def update_config(self, config: dict):
        """Updates config using a dictionary"""
        self.line_color = config.get('line_color', self.line_color)
        self.title = config.get('title', self.title)
        self.chart_type = config.get('chart_type', self.chart_type)
        self.width = config.get('width', self.width)
        self.height = config.get('height', self.height)
        self.float_position = config.get('float_position', self.float_position)
        self.margin = config.get('margin', self.margin)
        self.data = config.get('data', self.data)


        return self.get_chart_figure()

    def _create_line_chart(self):
        """Create a line chart from the data"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=self.data[self.y_column],
                    mode='lines+markers',
                    name=self.y_column,
                    line=dict(color=f'{self.line_color}', width=2)
                ))
            else:
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )

        return fig

    def _create_area_chart(self):
        """Create an area chart from the data"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=self.data[self.y_column],
                    mode='lines',
                    name=self.y_column,
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.3)',
                    line=dict(color=f'{self.line_color}', width=2)
                ))
            else:
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )

        return fig

    def get_chart_figure(self):
        """Get the appropriate chart figure based on chart_type"""
        if self.chart_type == 'line':
            return self._create_line_chart()
        elif self.chart_type == 'area':
            return self._create_area_chart()
        else:
            return self._create_bar_chart()

    def get_chart_component(self):
        """Get the complete Dash component with styling"""
        # Determine CSS float style
        float_style = {}
        if self.float_position:
            float_style['float'] = self.float_position

        style = {
            'width': self.width,
            'border': '2px solid #34495e',
            'padding': '10px',
            'margin': self.margin,
            **float_style
        }

        return html.Div([
            html.H4(self.title, style={'text-align': 'center', 'margin': '10px 0'}),
            dcc.Graph(
                id=self.chart_id,
                figure=self.get_chart_figure()
            )
        ], style=style)

    def update_data_source(self, data, y_column=None):
        """Update the data source and reload data"""
        self.data = data
        self.y_column = y_column if y_column else self.data.select_dtypes(include=[np.number], exclude=[datetime, 'object']).columns[0]

        return self.get_chart_figure()

    def change_y_column(self, new_y_column):
        """Change the y-axis columns_col"""
        self.y_column = new_y_column
        return self.get_chart_figure()

    def change_chart_type(self, new_chart_type):
        """Change the chart key_type"""
        self.chart_type = new_chart_type
        return self.get_chart_figure()

    def get_available_columns(self):
        """Get list of available columns in the loaded data"""
        if self.data is not None:
            return list(self.data.columns)
        return []

    def get_data_info(self):
        """Get basic information about the loaded data"""
        if self.data is not None:
            return {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'sample': self.data.head().to_dict()
            }
        return None

class MultiChart(FundamentalChart):
    """
    A multi-series chart component that can plot multiple data columns with different scales.

    Parameters:
    - chart_id: Unique identifier for the chart
    - TableClient: Client for accessing table data
    - keys: List of keys to fetch data from
    - title: Chart title
    - x_column: Column name for x-axis data (default: 'Date')
    - chart_type: 'bar', 'line', or 'area'
    - width: Width as percentage string or pixels
    - line_colors: List of colors for each series (optional)
    - height: Height in pixels (default: 300)
    - float_position: 'left', 'right', or None for CSS float positioning
    - margin: Margin string
    - dual_y: Whether to use dual y-axes (default: False)
    - secondary_y_columns: List of columns to plot on secondary y-axis
    """

    def __init__(self, chart_id,config=None, data=None, y_columns=None, title="Multi-Series Chart", x_column='date',
                 chart_type='line', width='49%', line_colors=['blue', 'red', 'green', 'purple'], height=300,
                 float_position='left', margin='5px 1%', dual_y=False, secondary_y_columns=None):

        # Initialize parent class without starting_key
        super().__init__(chart_id, title=title, x_column=x_column,
                         chart_type=chart_type, width=width, height=height,
                         float_position=float_position, margin=margin)
        self.config = config if config else {}

        self.line_colors = self.config.get('line_colors', line_colors)
        self.dual_y = config.get('dual_y', dual_y)
        self.secondary_y_columns = self.config.get('secondary_y_columns', secondary_y_columns)
        self.y_columns = config.get('y_columns', [])  # Columns to plot
        self.data = self.config.get('data', data)

        # Load data from multiple keys
        # Set default selected columns (all numeric columns)
        if self.data is not None and not self.y_columns:
            self.y_columns = [col for col in self.data.columns
                              if col != self.x_column and pd.api.types.is_numeric_dtype(self.data[col])]


    def set_selected_columns(self, columns):
        """Set which columns to plot"""
        if self.data is not None:
            self.y_columns = [col for col in self.data.columns if col in columns]
            self.config['y_columns'] = self.y_columns

        return self.get_chart_figure()

    def set_secondary_y_columns(self, columns):
        """Set which columns should be plotted on secondary y-axis"""
        self.secondary_y_columns = columns
        return self

    def enable_dual_y(self, enable=True):
        """Enable or disable dual y-axis"""
        self.dual_y = enable
        return self

    def _create_multi_line(self):
        """Create a multi-line chart with optional dual y-axis"""
        if self.dual_y and self.secondary_y_columns:
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        if self.data is not None and not self.data.empty and self.y_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.y_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    trace = go.Scatter(
                        x=x_data,
                        y=self.data[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )

                    # Add to secondary y-axis if specified
                    if self.dual_y and col in self.secondary_y_columns:
                        fig.add_trace(trace, secondary_y=True)
                    else:
                        if self.dual_y:
                            fig.add_trace(trace, secondary_y=False)
                        else:
                            fig.add_trace(trace)
        else:
            # No data or columns selected
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        # Update layout
        if self.dual_y and self.secondary_y_columns:
            # Set y-axes titles
            primary_cols = [col for col in self.y_columns if col not in self.secondary_y_columns]
            secondary_cols = [col for col in self.y_columns if col in self.secondary_y_columns]

            fig.update_yaxes(title_text=f"Primary: {', '.join(primary_cols)}", secondary_y=False)
            fig.update_yaxes(title_text=f"Secondary: {', '.join(secondary_cols)}", secondary_y=True)
            fig.update_xaxes(title_text=self.x_column)
        else:
            fig.update_layout(
                xaxis_title=self.x_column,
                yaxis_title="Value"
            )

        fig.update_layout(
            title=self.title,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='x unified',
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig

    def _create_multi_bar(self):
        """Create a multi-series bar chart"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.y_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.y_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=self.data[col],
                        name=col,
                        marker_color=color,
                        opacity=0.8
                    ))
        else:
            fig.add_trace(go.Bar(x=[], y=[], name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title="Value",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            barmode='group',  # Grouped bars
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )

        return fig

    def _create_multi_area(self):
        """Create a multi-series area chart (stacked)"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.y_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.y_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=self.data[col],
                        mode='lines',
                        name=col,
                        fill='tonexty' if i > 0 else 'tozeroy',
                        fillcolor=f'rgba({self._hex_to_rgb(color)}, 0.3)',
                        line=dict(color=color, width=2)
                    ))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title="Value",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )

        return fig

    def _create_normalized_chart(self):
        """Create a chart with normalized data (0-100 scale)"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.y_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.y_columns):
                if col in self.data.columns:
                    # Normalize data to 0-100 scale
                    col_data = self.data[col]
                    min_val = col_data.min()
                    max_val = col_data.max()
                    normalized_data = ((col_data - min_val) / (max_val - min_val)) * 100

                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=normalized_data,
                        mode='lines+markers',
                        name=f"{col} (Normalized)",
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ))

        fig.update_layout(
            title=f"{self.title} - Normalized (0-100)",
            xaxis_title=self.x_column,
            yaxis_title="Normalized Value (0-100)",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='x unified',
            template=self.theme,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB values"""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

    def get_chart_figure(self, normalized=False):
        """Get the appropriate chart figure based on chart_type"""
        if normalized:
            return self._create_normalized_chart()

        if self.chart_type == 'line':
            return self._create_multi_line()
        elif self.chart_type == 'area':
            return self._create_multi_area()
        elif self.chart_type == 'bar':
            return self._create_multi_bar()
        else:
            return self._create_multi_line()

    def get_chart_component(self, normalized=False):
        """Get the complete Dash component with styling"""
        float_style = {}
        if self.float_position:
            float_style['float'] = self.float_position

        style = {
            'width': self.width,
            'border': '2px solid #34495e',
            'padding': '10px',
            'margin': self.margin,
            **float_style
        }

        # Create columns_col selector dropdown
        column_options = []
        if self.data is not None:
            column_options = [{'label': col, 'value': col}
                              for col in self.data.columns
                              if col != self.x_column and pd.api.types.is_numeric_dtype(self.data[col])]

        return html.Div([
            html.H4(self.title, style={'text-align': 'center', 'margin': '10px 0'}),

            # Column selector
            html.Div([
                html.Label("Select Columns to Plot:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id=f'{self.chart_id}_column_selector',
                    options=column_options,
                    value=self.y_columns,
                    multi=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Chart key_type selector
            html.Div([
                html.Label("Chart Type:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.RadioItems(
                    id=f'{self.chart_id}_chart_type',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Area', 'value': 'area'}
                    ],
                    value=self.chart_type,
                    inline=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Options
            html.Div([
                dcc.Checklist(
                    id=f'{self.chart_id}_options',
                    options=[
                        {'label': 'Dual Y-Axis', 'value': 'dual_y'},
                        {'label': 'Normalized View', 'value': 'normalized'}
                    ],
                    value=['dual_y'] if self.dual_y else [],
                    inline=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Chart
            dcc.Graph(
                id=self.chart_id,
                figure=self.get_chart_figure(normalized=normalized)
            )
        ], style=style)

    def update_selected_columns(self, selected_columns):
        """Update selected columns and return new figure"""
        self.y_columns = selected_columns
        return self.get_chart_figure()

    def update_data_source(self, data, y_columns=None):
        self.data = data
        self.y_columns = y_columns if y_columns else self.data.select_dtypes(include=[np.number], exclude=['object', datetime]).columns.tolist()


    def update_chart_type(self, chart_type):
        """Update chart key_type and return new figure"""
        self.chart_type = chart_type
        return self.get_chart_figure()

    def update_options(self, options):
        """Update chart options and return new figure"""
        self.dual_y = 'dual_y' in options
        normalized = 'normalized' in options
        return self.get_chart_figure(normalized=normalized)

    def get_data_summary(self):
        """Get summary statistics for all numeric columns"""
        if self.data is not None:
            numeric_cols = [col for col in self.data.columns
                            if pd.api.types.is_numeric_dtype(self.data[col])]
            return self.data[numeric_cols].describe()
        return None

    def get_correlation_matrix(self):
        """Get correlation matrix for selected columns"""
        if self.data is not None and self.y_columns:
            return self.data[self.y_columns].corr()
        return None


class COTPlotter:
    """
    A class for plotting Commitment of Traders (COT) reports using Plotly.
    Creates interactive charts matching professional COT report visualizations.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the COT plotter with data."""
        self.df = df.copy()
        self.prepare_data()

        # Define color scheme
        self.colors = {
            'commercials': '#FF6B6B',
            'non_commercials': '#4ECDC4',
            'small_speculators': '#45B7D1',
            'swap_dealers': '#96CEB4',
            'money_managers': '#FFEAA7',
            'other_reportables': '#DDA0DD'
        }


    def prepare_data(self):
        """Prepare and clean the data for plotting."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')

        # Calculate net positions
        self.df['producer_merchant_net'] = (
            self.df['producer_merchant_processor_user_longs'] -
            self.df['producer_merchant_processor_user_shorts']
        )

        self.df['swap_dealer_net'] = (
            self.df['swap_dealer_longs'] -
            self.df['swap_dealer_shorts']
        )

        self.df['money_manager_net'] = (
            self.df['money_manager_longs'] -
            self.df['money_manager_shorts']
        )

        self.df['other_reportable_net'] = (
            self.df['other_reportable_longs'] -
            self.df['other_reportable_shorts']
        )

        # Calculate total non-commercial positions
        self.df['non_commercial_longs'] = (
            self.df['money_manager_longs'] + self.df['swap_dealer_longs']
        )

        self.df['non_commercial_shorts'] = (
            self.df['money_manager_shorts'] + self.df['swap_dealer_shorts']
        )

        self.df['non_commercial_net'] = (
            self.df['non_commercial_longs'] - self.df['non_commercial_shorts']
        )

    def plot_cot_report(self,
                       show_net_positions: bool = True,
                       show_disaggregated: bool = True,
                       date_range: Optional[Tuple[str, str]] = None,
                       title: str = "Commitment of Traders Report",
                       height: int = 800) -> go.Figure:
        """Create comprehensive COT report visualization."""

        # Filter data by date range if specified
        plot_data = self.df.copy()
        if date_range:
            start_date, end_date = date_range
            plot_data = plot_data[
                (plot_data['date'] >= start_date) &
                (plot_data['date'] <= end_date)
            ]

        # Create subplots
        if show_disaggregated:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Traditional COT View", "Disaggregated COT View"),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
        else:
            fig = go.Figure()

        # Plot traditional COT view
        self._add_traditional_cot(fig, plot_data, show_net_positions, row=1 if show_disaggregated else None)

        # Plot disaggregated view if requested
        if show_disaggregated:
            self._add_disaggregated_cot(fig, plot_data, show_net_positions, row=2)

        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Date",
            yaxis_title="Contracts"
        )

        if show_disaggregated:
            fig.update_yaxes(title_text="Contracts", row=1, col=1)
            fig.update_yaxes(title_text="Contracts", row=2, col=1)

        return fig

    def _add_traditional_cot(self, fig, data, show_net_positions, row=None):
        """Add traditional COT traces to figure."""
        dates = data['date']

        if show_net_positions:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_net'],
                          name='Commercials (Net)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_commercial_net'],
                          name='Non-Commercials (Net)', line=dict(color=self.colors['non_commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_reportable_longs'] - data['non_reportable_shorts'],
                          name='Small Speculators (Net)', line=dict(color=self.colors['small_speculators'], width=2))
            ]
        else:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_longs'],
                          name='Commercials (Long)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_shorts'],
                          name='Commercials (Short)', line=dict(color=self.colors['commercials'], width=2, dash='dash')),
                go.Scatter(x=dates, y=data['non_commercial_longs'],
                          name='Non-Commercials (Long)', line=dict(color=self.colors['non_commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_commercial_shorts'],
                          name='Non-Commercials (Short)', line=dict(color=self.colors['non_commercials'], width=2, dash='dash'))
            ]

        for trace in traces:
            fig.add_trace(trace, row=row, col=1)

        # Add zero line for net positions
        if show_net_positions:
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=row, col=1)

    def _add_disaggregated_cot(self, fig, data, show_net_positions, row):
        """Add disaggregated COT traces to figure."""
        dates = data['date']

        if show_net_positions:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_net'],
                          name='Producer/Merchant (Net)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['swap_dealer_net'],
                          name='Swap Dealers (Net)', line=dict(color=self.colors['swap_dealers'], width=2)),
                go.Scatter(x=dates, y=data['money_manager_net'],
                          name='Money Managers (Net)', line=dict(color=self.colors['money_managers'], width=2)),
                go.Scatter(x=dates, y=data['other_reportable_net'],
                          name='Other Reportables (Net)', line=dict(color=self.colors['other_reportables'], width=2))
            ]
        else:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_longs'],
                          name='Producer/Merchant (Long)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['swap_dealer_longs'],
                          name='Swap Dealers (Long)', line=dict(color=self.colors['swap_dealers'], width=2)),
                go.Scatter(x=dates, y=data['money_manager_longs'],
                          name='Money Managers (Long)', line=dict(color=self.colors['money_managers'], width=2)),
                go.Scatter(x=dates, y=data['other_reportable_longs'],
                          name='Other Reportables (Long)', line=dict(color=self.colors['other_reportables'], width=2))
            ]

        for trace in traces:
            fig.add_trace(trace, row=row, col=1)

        # Add zero line for net positions
        if show_net_positions:
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=row, col=1)

    def plot_concentration_analysis(self,
                                  date_range: Optional[Tuple[str, str]] = None,
                                  height: int = 600) -> go.Figure:
        """Plot concentration analysis as stacked area chart."""

        # Filter data by date range if specified
        plot_data = self.df.copy()
        if date_range:
            start_date, end_date = date_range
            plot_data = plot_data[
                (plot_data['date'] >= start_date) &
                (plot_data['date'] <= end_date)
            ]

        # Calculate percentages
        total_longs = plot_data['total_reportable_longs'] + plot_data['non_reportable_longs']

        fig = go.Figure()

        # Add stacked area traces
        categories = [
            ('Commercials', plot_data['producer_merchant_processor_user_longs'] / total_longs * 100, self.colors['commercials']),
            ('Swap Dealers', plot_data['swap_dealer_longs'] / total_longs * 100, self.colors['swap_dealers']),
            ('Money Managers', plot_data['money_manager_longs'] / total_longs * 100, self.colors['money_managers']),
            ('Other Reportables', plot_data['other_reportable_longs'] / total_longs * 100, self.colors['other_reportables']),
            ('Non-Reportables', plot_data['non_reportable_longs'] / total_longs * 100, self.colors['small_speculators'])
        ]

        for name, values, color in categories:
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=values,
                mode='lines',
                stackgroup='one',
                name=name,
                line=dict(width=0.5, color=color),
                fill='tonexty'
            ))

        fig.update_layout(
            title='COT Position Concentration Analysis',
            xaxis_title='Date',
            yaxis_title='Percentage of Total Open Interest (%)',
            height=height,
            hovermode='x unified'
        )

        return fig

    def get_latest_positions(self) -> Dict:
        """Get the latest position data for all categories."""
        latest_data = self.df.iloc[-1]

        return {
            'date': latest_data['date'],
            'commercials': {
                'longs': latest_data['producer_merchant_processor_user_longs'],
                'shorts': latest_data['producer_merchant_processor_user_shorts'],
                'net': latest_data['producer_merchant_net']
            },
            'swap_dealers': {
                'longs': latest_data['swap_dealer_longs'],
                'shorts': latest_data['swap_dealer_shorts'],
                'net': latest_data['swap_dealer_net']
            },
            'money_managers': {
                'longs': latest_data['money_manager_longs'],
                'shorts': latest_data['money_manager_shorts'],
                'net': latest_data['money_manager_net']
            }
        }

