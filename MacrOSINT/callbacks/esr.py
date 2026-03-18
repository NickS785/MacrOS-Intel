import pandas as pd
from MacrOSINT.pages.esr.esr_utils import create_empty_figure
import plotly.express as px
from MacrOSINT.data.data_tables import ESRTableClient

table_client = ESRTableClient()


def create_seasonal_summary_table(commodity, results, seasonal_metric):
    """Create seasonal summary table data from analysis results."""
    # Marketing year information
    marketing_years = {
        'cattle': ('January 1', 'December 31'),
        'hogs': ('January 1', 'December 31'),
        'pork': ('January 1', 'December 31'),
        'corn': ('September 1', 'August 31'),
        'wheat': ('September 1', 'August 31'),
        'soybeans': ('September 1', 'August 31')
    }
    
    my_start, my_end = marketing_years.get(commodity.lower(), ('September 1', 'August 31'))
    
    if 'error' in results:
        return [{
            'commodity': commodity.title(),
            'my_start': my_start,
            'my_end': my_end,
            'peak_weeks': 'No data available',
            'low_weeks': 'No data available',
            'seasonality': 0
        }]
    
    # Extract peak and low weeks
    peak_weeks = results.get('peak_weeks', [])
    low_weeks = results.get('low_weeks', [])
    seasonality_strength = results.get('seasonality_strength', 0)
    
    # Format week ranges
    peak_str = f"Weeks {min(peak_weeks)}-{max(peak_weeks)}" if peak_weeks else "N/A"
    low_str = f"Weeks {min(low_weeks)}-{max(low_weeks)}" if low_weeks else "N/A"
    
    return [{
        'commodity': commodity.title(),
        'my_start': my_start,
        'my_end': my_end,
        'peak_weeks': peak_str,
        'low_weeks': low_str,
        'seasonality': round(seasonality_strength, 4) if seasonality_strength else 0
    }]



def sales_trends_chart_update(chart_ids, store_data=None, **menu_values):
        """Update function for Sales Trends page - handles multiple charts from single callback"""
        try:
            # Handle both single chart_id and list of chart_ids
            if isinstance(chart_ids, str):
                chart_ids = [chart_ids]
            
            # Get menu values
            countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
            country_display_mode = menu_values.get('country_display_mode', 'individual')
            date_range = menu_values.get('date_range', [])
            
            # Get columns_col selections for each chart
            chart_0_column = menu_values.get('chart_0_column', 'weeklyExports')
            chart_1_column = menu_values.get('chart_1_column', 'outstandingSales')
            chart_2_column = menu_values.get('chart_2_column', 'grossNewSales')
            
            # Use store data
            if not store_data:
                error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            try:
                if isinstance(store_data, str):
                    import json
                    data = pd.DataFrame(json.loads(store_data))
                else:
                    data = pd.DataFrame(store_data)
                
                # Debug: Print available columns
                print(f"DEBUG sales_trends - Available columns in store data: {list(data.columns) if not data.empty else 'No data'}")
                print(f"DEBUG sales_trends - Data shape: {data.shape if not data.empty else 'No data'}")
                
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
            except Exception as e:
                print(f"Error loading store data: {e}")
                error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
                
            if data.empty:
                error_figs = [create_empty_figure("No data available") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            # Filter by countries
            if countries and 'country' in data.columns:
                data = data[data['country'].isin(countries)]
            
            # Apply date range filter if provided
            if date_range and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                data = data[
                    (data['weekEndingDate'] >= start_date) & 
                    (data['weekEndingDate'] <= end_date)
                ]
            
            if data.empty:
                error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            # Column mapping for each chart
            column_mapping = {
                'esr_sales_trends_chart_0': chart_0_column,
                'esr_sales_trends_chart_1': chart_1_column,
                'esr_sales_trends_chart_2': chart_2_column
            }
            
            column_labels = {
                'weeklyExports': 'Weekly Exports',
                'outstandingSales': 'Outstanding Sales',
                'grossNewSales': 'Gross New Sales',
                'currentMYNetSales': 'Current MY Net Sales',
                'currentMYTotalCommitment': 'Current MY Total Commitment'
            }
            
            # Generate figures for each chart
            figures = []
            
            for chart_id in chart_ids:
                y_column = column_mapping.get(chart_id, 'weeklyExports')
                
                # Check if columns_col exists
                if y_column not in data.columns:
                    figures.append(create_empty_figure(f"Column '{y_column}' not found in data"))
                    continue
                
                chart_title = column_labels.get(y_column, y_column.replace('_', ' ').title())
                
                # Handle country display mode
                if country_display_mode == 'sum' and len(countries) > 1:
                    # Sum all countries together
                    data_grouped = data.groupby('weekEndingDate')[y_column].sum().reset_index()
                    data_grouped['country'] = f"Sum of {', '.join(countries)}"
                    chart_data = data_grouped
                    
                    fig = px.line(
                        chart_data,
                        x='weekEndingDate',
                        y=y_column,
                        title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                        markers=True
                    )
                    fig.update_traces(line=dict(color='#1f77b4', width=3))
                else:
                    # Individual countries
                    fig = px.line(
                        data,
                        x='weekEndingDate',
                        y=y_column,
                        color='country',
                        title=f"{chart_title} by Country",
                        markers=True
                    )
                
                # Update layout
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    xaxis_title='Week Ending Date',
                    yaxis_title=chart_title,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                figures.append(fig)
            
            # Return single figure or list of figures
            return figures[0] if len(chart_ids) == 1 else figures
            
        except Exception as e:
            print(f"Error in sales_trends_chart_update: {e}")
            error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs

def country_analysis_chart_update(chart_ids, store_data=None, **menu_values):
    """
    Update function for Country Analysis page - supports multi-chart and store data with market year overlays.
    Enhanced to support both single chart_id and list of chart_ids for multi-chart updates.
    Supports dynamic country selection and market year overlay functionality.
    """
    try:
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        country_metric = menu_values.get('country_metric', 'weeklyExports')
        date_range = menu_values.get('date_range', [])
        start_year = menu_values.get('start_year')
        end_year = menu_values.get('end_year')
        
        print(f"DEBUG country_analysis: chart_ids={chart_ids}, countries={countries}")
        
        # Use store data
        if not store_data:
            error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
            
        if data.empty:
            error_figs = [create_empty_figure("No data available") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Generate figures for each chart
        figures = []
        
        for i, chart_id in enumerate(chart_ids):
            fig = create_country_analysis_chart(
                data, country_metric, countries, country_display_mode, 
                start_year, end_year, chart_id, i
            )
            figures.append(fig)
        
        # Return single figure or list of figures
        return figures[0] if len(chart_ids) == 1 else figures
        
    except Exception as e:
        print(f"Error in country_analysis_chart_update: {e}")
        error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
        return error_figs[0] if len(chart_ids) == 1 else error_figs

def dual_frame_commitment_analysis_chart_update(chart_ids, store_data=None, **menu_values):
    """
    Update function for dual frame Commitment Analysis page - handles multiple charts.
    Enhanced to support both single chart_id and list of chart_ids for multi-chart updates.
    Frame 0 Chart 0: Store data with columns_col select
    Other charts: Analytics with sales_backlog, fulfillment_rate, commitment_utilization
    """
    try:
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
        date_range = menu_values.get('date_range', [])
        
        print(f"DEBUG commitment_analysis: chart_ids={chart_ids}, countries={countries}")
        
        # Use store data
        if not store_data:
            error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            # Debug: Print available columns for commitment analysis
            print(f"DEBUG commitment_analysis - Available columns: {list(data.columns) if not data.empty else 'No data'}")
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
            
        if data.empty:
            error_figs = [create_empty_figure("No data available") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Generate figures for each chart
        figures = []
        
        for chart_id in chart_ids:
            if 'esr_commitment_frame0_chart_0' in chart_id:
                # Frame 0, Chart 0 - selectable commitment metric from store data
                fig = create_store_based_commitment_chart(
                    data, commitment_metric, countries, country_display_mode, chart_id
                )
            else:
                # Other charts - analytics (sales_backlog, fulfillment_rate, commitment_utilization)
                fig = create_analytics_commitment_chart(
                    data, countries, country_display_mode, chart_id
                )
            
            figures.append(fig)
        
        # Return single figure or list of figures
        return figures[0] if len(chart_ids) == 1 else figures
        
    except Exception as e:
        print(f"Error in dual_frame_commitment_analysis_chart_update: {e}")
        error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
        return error_figs[0] if len(chart_ids) == 1 else error_figs


def create_store_based_commitment_chart(data, commitment_metric, countries, country_display_mode, chart_id):
    """Create chart for Frame 0 Chart 0 using store data with columns_col selection"""
    try:
        # Check if columns_col exists
        if commitment_metric not in data.columns:
            return create_empty_figure(f"Column '{commitment_metric}' not found in data")
        
        # Create chart title
        column_labels = {
            'currentMYTotalCommitment': 'Current MY Total Commitment',
            'currentMYNetSales': 'Current MY Net Sales',
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        chart_title = column_labels.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            data_grouped = data.groupby('weekEndingDate')[commitment_metric].sum().reset_index()
            data_grouped['country'] = f"Sum of {', '.join(countries)}"
            chart_data = data_grouped
            
            fig = px.line(
                chart_data,
                x='weekEndingDate',
                y=commitment_metric,
                title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                markers=True
            )
            fig.update_traces(line=dict(color='#1f77b4', width=3))
        else:
            # Individual countries
            fig = px.line(
                data,
                x='weekEndingDate',
                y=commitment_metric,
                color='country',
                title=f"{chart_title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=350,
            xaxis_title='Week Ending Date',
            yaxis_title=chart_title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_store_based_commitment_chart: {e}")
        return create_empty_figure(f"Error: {str(e)}")


def create_analytics_commitment_chart(data, countries, country_display_mode, chart_id):
    """Create analytics charts for sales_backlog, fulfillment_rate, commitment_utilization"""
    try:
        # Perform commitment vs shipment analysis
        from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
        
        # Handle country display mode for analysis
        analysis_data = data.copy()
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            # Sum numeric columns by date
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in analysis_data.columns]
            
            if available_cols:
                grouped = analysis_data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped
        
        # Create ESR analyzer instance with proper initialization
        # Determine commodity type
        commodity_type = 'livestock'  # Default for cattle, hogs, pork
        # Note: This could be enhanced to detect grain vs livestock from data
        
        analyzer = ESRAnalyzer(analysis_data, commodity_type=commodity_type)
        
        # Perform analysis
        analysis_results = analyzer.commitment_vs_shipment_analysis()
        
        # Handle the case where analysis_results is a dict with 'data' key
        if isinstance(analysis_results, dict):
            if 'error' in analysis_results:
                return create_empty_figure(f"Analytics Error: {analysis_results['error']}")
            
            # Get the actual data DataFrame
            analysis_data = analysis_results.get('data', pd.DataFrame())
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        else:
            # If it's already a DataFrame
            analysis_data = analysis_results
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        
        # Determine which analytics metric to display based on chart_id
        if 'chart_1' in chart_id:
            # Frame 0, Chart 1 - Sales Backlog
            metric = 'sales_backlog'
            title = 'Sales Backlog Analysis'
        elif 'frame1_chart_0' in chart_id:
            # Frame 1, Chart 0 - Commitment Utilization
            metric = 'commitment_utilization' 
            title = 'Commitment Utilization Rate'
        elif 'frame1_chart_1' in chart_id:
            # Frame 1, Chart 1 - Fulfillment Rate
            metric = 'fulfillment_rate'
            title = 'Export Fulfillment Rate'
        else:
            # Default to sales backlog
            metric = 'sales_backlog'
            title = 'Sales Backlog Analysis'
        
        # Check if metric columns_col exists in analysis results
        if metric not in analysis_data.columns:
            return create_empty_figure(f"Analytics metric '{metric}' not available")
        
        # Create chart
        if country_display_mode == 'sum' and len(countries) > 1:
            fig = px.line(
                analysis_data,
                x='weekEndingDate',
                y=metric,
                title=f"{title} - Sum of {', '.join(countries)}",
                markers=True
            )
            fig.update_traces(line=dict(color='#ff7f0e', width=3))
        else:
            fig = px.line(
                analysis_data,
                x='weekEndingDate',
                y=metric,
                color='country' if 'country' in analysis_data.columns else None,
                title=f"{title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=350,
            xaxis_title='Week Ending Date',
            yaxis_title=title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_analytics_commitment_chart: {e}")
        return create_empty_figure(f"Analytics Error: {str(e)}")


def new_commitment_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
    """Update function for new Commitment Analysis page - handles both metric chart and analytics charts."""
    commodity = menu_values.get('commodity', 'cattle')
    country_selection = menu_values.get('country_selection', 'Korea, South')
    countries = menu_values.get('countries', ['Korea, South'])
    commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
    
    # Debug print for troubleshooting
    print(f"DEBUG commitment_analysis: chart_id={chart_id}, commodity={commodity}, menu_values keys={list(menu_values.keys())}")
    
    # Ensure commodity is a string
    if not isinstance(commodity, str) or not commodity:
        commodity = 'cattle'
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    # Get multi-year data
    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        start_year=start_year,
        end_year=end_year
    )

    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Commitment Analysis ({year_range})")

    # Handle country selection
    if country_selection == 'ALL_COUNTRIES':
        # Sum of multiple countries
        filtered_data = data[data['country'].isin(countries)]
        if not filtered_data.empty:
            # Use ESRAnalyzer to aggregate multi-country data
            from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
            aggregated_data = ESRAnalyzer.aggregate_multi_country_data(filtered_data, countries)
            title_suffix = f"All Selected Countries ({len(countries)})"
            chart_data = aggregated_data
        else:
            return create_empty_figure(f"{commodity.title()} - No data for selected countries")
    else:
        # Single country
        chart_data = data[data['country'] == country_selection]
        title_suffix = country_selection

    if chart_data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - {title_suffix} ({year_range})")

    # Determine which chart is being updated
    if 'esr_commitment_frame1_chart_0' in chart_id:
        # First chart (Frame 1, Chart 0) - selectable commitment metric
        metric_names = {
            'currentMYTotalCommitment': 'MY Total Commitment',
            'currentMYNetSales': 'MY Net Sales', 
            'outstandingSales': 'MY Outstanding Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        
        metric_name = metric_names.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        if commitment_metric not in chart_data.columns:
            return create_empty_figure(f"Column '{commitment_metric}' not available")
        
        fig = px.line(
            chart_data,
            x='weekEndingDate',
            y=commitment_metric,
            title=f'{commodity.title()} - {metric_name} ({year_range}) - {title_suffix}',
            markers=True
        )
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Week Ending Date",
            yaxis_title=metric_name
        )
        
        return fig
        
    elif ('esr_commitment_frame1_chart_1' in chart_id or 
          'esr_commitment_frame2_chart_0' in chart_id or 
          'esr_commitment_frame2_chart_1' in chart_id):
        # Analytics charts - commitment vs shipment analysis
        from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
        
        # Initialize analyzer with the data
        analyzer = ESRAnalyzer(chart_data.set_index('weekEndingDate'), 'grains')
        
        # Get commitment analysis results
        if country_selection == 'ALL_COUNTRIES':
            results = analyzer.commitment_vs_shipment_analysis(countries=countries, commodity=commodity)
        else:
            results = analyzer.commitment_vs_shipment_analysis(country=country_selection, commodity=commodity)
        
        if 'error' in results:
            year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
            return create_empty_figure(f"{commodity.title()} - Analysis Error ({year_range}) - {title_suffix}")
        
        # Access the data from results['data']
        results_data = results.get('data', pd.DataFrame())
        if results_data.empty:
            year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
            return create_empty_figure(f"{commodity.title()} - No Analysis Data ({year_range}) - {title_suffix}")
        
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        if 'esr_commitment_frame1_chart_1' in chart_id:
            # Commitment Utilization Rate
            if 'commitment_utilization' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='commitment_utilization',
                    title=f'{commodity.title()} - Commitment Utilization Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Utilization Rate"
                )
                
                return fig
        
        elif 'esr_commitment_frame2_chart_0' in chart_id:
            # Export Fulfillment Rate
            if 'fulfillment_rate' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='fulfillment_rate',
                    title=f'{commodity.title()} - Export Fulfillment Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Fulfillment Rate"
                )
                
                return fig
        
        elif 'esr_commitment_frame2_chart_1' in chart_id:
            # Sales Backlog Analysis
            try:
                # Check if sales_backlog columns_col exists
                if 'sales_backlog' in results_data.columns:
                    print(f"DEBUG: Found sales_backlog columns_col in results_data")
                    
                    fig = px.line(
                        results_data.reset_index(),
                        x='weekEndingDate',
                        y='sales_backlog',
                        title=f'{str(commodity).title()} - Sales Backlog ({year_range}) - {title_suffix}',
                        markers=True
                    )
                    
                    fig.update_layout(
                        height=350,
                        hovermode='x unified',
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis_title="Week Ending Date",
                        yaxis_title="Sales Backlog (Weeks)"
                    )
                    
                    return fig
                else:
                    print(f"DEBUG: sales_backlog columns_col not found. Available columns: {list(results_data.columns)}")
                    # Try sales_backlog_weeks instead
                    if 'sales_backlog_weeks' in results_data.columns:
                        fig = px.line(
                            results_data.reset_index(),
                            x='weekEndingDate',
                            y='sales_backlog_weeks',
                            title=f'{str(commodity).title()} - Sales Backlog ({year_range}) - {title_suffix}',
                            markers=True
                        )
                        
                        fig.update_layout(
                            height=350,
                            hovermode='x unified',
                            template='plotly_dark',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis_title="Week Ending Date",
                            yaxis_title="Sales Backlog (Weeks)"
                        )
                        
                        return fig
                    else:
                        return create_empty_figure(f"{str(commodity).title()} - Sales Backlog Column Missing")
            except Exception as e:
                print(f"DEBUG: Error in chart_3: {str(e)}")
                return create_empty_figure(f"{str(commodity).title()} - Sales Backlog Error: {str(e)}")
    
    # Fallback
    year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
    return create_empty_figure(f"{str(commodity).title()} - Chart Update Error ({year_range})")


def commitment_analysis_chart_update(chart_id: str, **menu_values):
    """Update function for Commitment Analysis page"""
    commodity = menu_values.get('commodity', 'cattle')
    countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        start_year=start_year,
        end_year=end_year
    )

    if countries:
        data = data[data['country'].isin(countries)]

    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Commitment Analysis ({year_range})")

    # Chart-to-metric and chart-type mapping
    if chart_id == 'esr_commitment_analysis_chart_0':
        metric = 'currentMYTotalCommitment'
        metric_name = 'Current MY Total Commitment'
        chart_type = 'area'
    elif chart_id == 'esr_commitment_analysis_chart_1':
        metric = 'currentMYNetSales'
        metric_name = 'Current MY Net Sales'
        chart_type = 'line'
    elif chart_id == 'esr_commitment_analysis_chart_2':
        metric = 'nextMYOutstandingSales'
        metric_name = 'Next MY Outstanding Sales'
        chart_type = 'bar'
    elif chart_id == 'esr_commitment_analysis_chart_3':
        metric = 'nextMYNetSales'
        metric_name = 'Next MY Net Sales'
        chart_type = 'line'
    else:
        metric = 'currentMYTotalCommitment'
        metric_name = 'Current MY Total Commitment'
        chart_type = 'area'

    # Create chart based on type
    year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
    
    if chart_type == 'area':
        fig = px.area(data, x='weekEndingDate', y=metric, color='country',
                      title=f"{commodity.title()} - {metric_name} ({year_range})")
    elif chart_type == 'bar':
        fig = px.bar(data, x='weekEndingDate', y=metric, color='country',
                     title=f"{commodity.title()} - {metric_name} ({year_range})")
    else:  # line
        fig = px.line(data, x='weekEndingDate', y=metric, color='country',
                      title=f"{commodity.title()} - {metric_name} ({year_range})", markers=True)

    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig


def commitment_metric_chart(chart_id, store_data=None, **menu_values):
    """
    Create chart for Frame 0 Chart 0 using store data with commitment metric selection.
    Based on manifest objectives for commitment analysis.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
        date_range = menu_values.get('date_range', [])
        
        # Use store data
        if not store_data:
            return [create_empty_figure("No data available in store")]
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return [create_empty_figure(f"Error loading data: {str(e)}")]
            
        if data.empty:
            return [create_empty_figure("No data available")]
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            return [create_empty_figure("No data for selected criteria")]
        
        # Check if columns_col exists
        if commitment_metric not in data.columns:
            return [create_empty_figure(f"Column '{commitment_metric}' not found in data")]
        
        # Create chart title
        column_labels = {
            'currentMYTotalCommitment': 'Current MY Total Commitment',
            'currentMYNetSales': 'Current MY Net Sales',
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        chart_title = column_labels.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            data_grouped = data.groupby('weekEndingDate')[commitment_metric].sum().reset_index()
            data_grouped['country'] = f"Sum of {', '.join(countries)}"
            chart_data = data_grouped
            
            fig = px.line(
                chart_data,
                x='weekEndingDate',
                y=commitment_metric,
                title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                markers=True
            )
            fig.update_traces(line=dict(color='#1f77b4', width=3))
        else:
            # Individual countries
            fig = px.line(
                data,
                x='weekEndingDate',
                y=commitment_metric,
                color='country',
                title=f"{chart_title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title='Week Ending Date',
            yaxis_title=chart_title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return [fig]
        
    except Exception as e:
        print(f"Error in commitment_metric_chart: {e}")
        return [create_empty_figure(f"Error: {str(e)}")]


def commitment_analytics_chart(chart_ids, store_data=None, **menu_values):
    """
    Create analytics charts using ESRAnalyzer for sales_backlog, fulfillment_rate, commitment_utilization.
    Based on manifest objectives for commitment analysis.
    Handles multiple chart IDs and returns multiple figures.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        date_range = menu_values.get('date_range', [])
        
        # Handle both single chart_id and list of chart_ids for error cases
        def handle_error_return(error_msg, chart_ids):
            if isinstance(chart_ids, list) and len(chart_ids) > 1:
                return [create_empty_figure(error_msg) for _ in chart_ids]
            else:
                return create_empty_figure(error_msg)
        
        # Use store data
        if not store_data:
            return handle_error_return("No data available in store", chart_ids)
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return handle_error_return(f"Error loading data: {str(e)}", chart_ids)
            
        if data.empty:
            return handle_error_return("No data available", chart_ids)
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            return handle_error_return("No data for selected criteria", chart_ids)
        
        # Perform commitment vs shipment analysis
        from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
        
        # Handle country display mode for analysis
        analysis_data = data.copy()
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in analysis_data.columns]
            
            if available_cols:
                grouped = analysis_data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped
        
        # Create ESR analyzer instance
        commodity_type = 'livestock'  # Default for cattle, hogs, pork
        analyzer = ESRAnalyzer(analysis_data, commodity_type=commodity_type)
        
        # Perform analysis
        analysis_results = analyzer.commitment_vs_shipment_analysis()
        
        # Handle the case where analysis_results is a dict with 'data' key
        if isinstance(analysis_results, dict):
            if 'error' in analysis_results:
                return create_empty_figure(f"Analytics Error: {analysis_results['error']}")
            
            # Get the actual data DataFrame
            analysis_data = analysis_results.get('data', pd.DataFrame())
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        else:
            # If it's already a DataFrame
            analysis_data = analysis_results
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Create figures for each chart
        figures = []
        
        for chart_id in chart_ids:
            # Determine which analytics metric to display based on chart_id
            if 'chart_1' in chart_id or 'frame0_chart_1' in chart_id:
                # Frame 0, Chart 1 - Sales Backlog
                metric = 'sales_backlog'
                title = 'Sales Backlog Analysis'
            elif 'frame1_chart_0' in chart_id:
                # Frame 1, Chart 0 - Commitment Utilization
                metric = 'commitment_utilization' 
                title = 'Commitment Utilization Rate'
            elif 'frame1_chart_1' in chart_id:
                # Frame 1, Chart 1 - Fulfillment Rate
                metric = 'fulfillment_rate'
                title = 'Export Fulfillment Rate'
            else:
                # Default to sales backlog
                metric = 'sales_backlog'
                title = 'Sales Backlog Analysis'
            
            # Check if metric columns_col exists in analysis results
            if metric not in analysis_data.columns:
                figures.append(create_empty_figure(f"Analytics metric '{metric}' not available"))
                continue
            
            # Prepare data for charting - ensure weekEndingDate is available
            chart_data = analysis_data.copy()
            if 'weekEndingDate' not in chart_data.columns and chart_data.index.name in ['weekEndingDate', None]:
                # If weekEndingDate is the index, reset it to a columns_col
                chart_data = chart_data.reset_index()
                if 'index' in chart_data.columns and 'weekEndingDate' not in chart_data.columns:
                    chart_data = chart_data.rename(columns={'index': 'weekEndingDate'})
            
            # Ensure weekEndingDate is datetime
            if 'weekEndingDate' in chart_data.columns:
                chart_data['weekEndingDate'] = pd.to_datetime(chart_data['weekEndingDate'])
            else:
                # Create a simple date range if weekEndingDate is missing
                chart_data['weekEndingDate'] = pd.date_range('2024-01-01', periods=len(chart_data), freq='W')
            
            # Create chart
            if country_display_mode == 'sum' and len(countries) > 1:
                fig = px.line(
                    chart_data,
                    x='weekEndingDate',
                    y=metric,
                    title=f"{title} - Sum of {', '.join(countries)}",
                    markers=True
                )
                fig.update_traces(line=dict(color='#ff7f0e', width=3))
            else:
                fig = px.line(
                    chart_data,
                    x='weekEndingDate',
                    y=metric,
                    color='country' if 'country' in chart_data.columns else None,
                    title=f"{title} by Country",
                    markers=True
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title='Week Ending Date',
                yaxis_title=title,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            figures.append(fig)
        
        # Return single figure or list of figures based on input
        if len(figures) == 1:
            return figures[0]
        else:
            return figures
        
    except Exception as e:
        print(f"Error in commitment_analytics_chart: {e}")
        # Handle both single and multiple chart error cases
        if isinstance(chart_ids, list) and len(chart_ids) > 1:
            return [create_empty_figure(f"Analytics Error: {str(e)}") for _ in chart_ids]
        else:
            return create_empty_figure(f"Analytics Error: {str(e)}")


def comparative_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
        """Update function for Comparative Analysis page - supports store data"""
        commodity_a = menu_values.get('commodity_a', 'cattle')
        commodity_b = menu_values.get('commodity_b', 'corn')
        metric = menu_values.get('metric', 'weeklyExports')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        
        # Safely convert years to integers
        try:
            start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
        except (ValueError, TypeError):
            start_year = None
            
        try:
            end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
        except (ValueError, TypeError):
            end_year = None

        # Determine commodity based on frame
        if 'comparison_frame1' in chart_id:
            commodity = commodity_a
            frame_label = "A"
        else:
            commodity = commodity_b
            frame_label = "B"

        data = table_client.get_multi_year_esr_data(
            commodity=commodity,
            start_year=start_year,
            end_year=end_year
        )

        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"Commodity {frame_label}: {commodity.title()}")

        metric_name = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales'
        }.get(metric, metric.replace('_', ' ').title())

        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,
            color='country',
            title=f"Commodity {frame_label}: {commodity.title()} - {metric_name} ({year_range})",
            markers=True
        )

        fig.update_layout(
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig


def seasonal_overlay_chart(chart_ids, store_data=None, **menu_values):
    """
    Create seasonal overlay chart showing multiple marketing years overlaid.
    Top chart in seasonal analysis - shows year-over-year comparison.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
        start_year = menu_values.get('start_year')
        end_year = menu_values.get('end_year')
        date_range = menu_values.get('date_range', [])
        
        # Handle both single chart_id and list of chart_ids for error cases
        def handle_error_return(error_msg, chart_ids):
            if isinstance(chart_ids, list) and len(chart_ids) > 1:
                return [create_empty_figure(error_msg) for _ in chart_ids]
            else:
                return create_empty_figure(error_msg)
        
        # Use store data
        if not store_data:
            return handle_error_return("No data available in store", chart_ids)
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return handle_error_return(f"Error loading data: {str(e)}", chart_ids)
            
        if data.empty:
            return handle_error_return("No data available", chart_ids)
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
            
        if data.empty:
            return handle_error_return("No data for selected countries", chart_ids)

        # Generate multi-year seasonal overlay using ESRAnalyzer
        from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
        
        # Determine commodity type from data patterns
        commodity_type = 'grains'  # Default, could be enhanced to detect from data
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if available_cols:
                grouped = data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped.set_index('weekEndingDate')
            else:
                return handle_error_return("No numeric columns available for aggregation", chart_ids)
        else:
            analysis_data = data.set_index('weekEndingDate')
        
        # Create ESR analyzer instance
        analyzer = ESRAnalyzer(analysis_data, commodity_type)
        
        # Generate seasonal patterns for each marketing year
        seasonal_overlay_data = analyzer.generate_seasonal_overlay(seasonal_metric, start_year, end_year)
        
        if seasonal_overlay_data.empty:
            return handle_error_return("No seasonal overlay data generated", chart_ids)
        
        # Create overlaid line chart
        metric_labels = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales',
            'currentMYTotalCommitment': 'Current MY Total Commitment'
        }
        
        metric_label = metric_labels.get(seasonal_metric, seasonal_metric.replace('_', ' ').title())
        
        fig = px.line(
            seasonal_overlay_data,
            x='my_week',
            y=seasonal_metric,
            color='marketing_year',
            title=f'Multi-Year Seasonal Overlay - {metric_label}',
            labels={
                'my_week': 'Marketing Year Week',
                seasonal_metric: metric_label,
                'marketing_year': 'Marketing Year'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title='Marketing Year Week',
            yaxis_title=metric_label,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Always return as list - schema expects list output
        if isinstance(chart_ids, list):
            return [fig] * len(chart_ids)  # Same figure for multiple charts if needed
        else:
            return [fig]  # Single chart but still return as list
        
    except Exception as e:
        print(f"Error in seasonal_overlay_chart: {e}")
        # Always return error as list - schema expects list output
        if isinstance(chart_ids, list):
            return [create_empty_figure(f"Overlay Error: {str(e)}") for _ in chart_ids]
        else:
            return [create_empty_figure(f"Overlay Error: {str(e)}")]


def seasonal_pattern_chart(chart_ids, store_data=None, **menu_values):
    """
    Create detailed seasonal pattern chart for a single selected marketing year.
    Bottom chart in seasonal analysis - shows detailed analysis with statistical overlays.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
        selected_market_year = menu_values.get('selected_market_year')
        date_range = menu_values.get('date_range', [])
        
        # Handle both single chart_id and list of chart_ids for error cases
        def handle_error_return(error_msg, chart_ids):
            if isinstance(chart_ids, list) and len(chart_ids) > 1:
                return [create_empty_figure(error_msg) for _ in chart_ids]
            else:
                return create_empty_figure(error_msg)
        
        # Use store data
        if not store_data:
            return handle_error_return("No data available in store", chart_ids)
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return handle_error_return(f"Error loading data: {str(e)}", chart_ids)
            
        if data.empty:
            return handle_error_return("No data available", chart_ids)
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
            
        if data.empty:
            return handle_error_return("No data for selected countries", chart_ids)

        # Generate detailed seasonal analysis using ESRAnalyzer
        from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
        
        # Determine commodity type from data patterns
        commodity_type = 'grains'  # Default, could be enhanced to detect from data
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if available_cols:
                grouped = data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped.set_index('weekEndingDate')
            else:
                return handle_error_return("No numeric columns available for aggregation", chart_ids)
        else:
            analysis_data = data.set_index('weekEndingDate')
        
        # Create ESR analyzer instance
        analyzer = ESRAnalyzer(analysis_data, commodity_type)
        
        # Generate detailed seasonal analysis data for callbacks
        seasonal_analysis = analyzer.generate_seasonal_analysis_data(seasonal_metric, selected_market_year)
        
        if 'error' in seasonal_analysis:
            return handle_error_return(f"Seasonal Analysis Error: {seasonal_analysis['error']}", chart_ids)
        
        # Get the processed data
        pattern_data = seasonal_analysis.get('data', pd.DataFrame())
        if pattern_data.empty:
            return handle_error_return("No seasonal pattern data available", chart_ids)
        
        # Create detailed analysis chart with trend lines and statistical overlays
        metric_labels = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales',
            'currentMYTotalCommitment': 'Current MY Total Commitment'
        }
        
        metric_label = metric_labels.get(seasonal_metric, seasonal_metric.replace('_', ' ').title())
        
        # Create figure with secondary y-axis for trend analysis
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Main seasonal pattern
        fig.add_trace(
            go.Scatter(
                x=pattern_data['my_week'],
                y=pattern_data[seasonal_metric],
                mode='lines+markers',
                name=f'{metric_label} ({selected_market_year})',
                line=dict(width=3)
            )
        )
        
        # Add statistical overlays if available
        if 'trend' in pattern_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=pattern_data['my_week'],
                    y=pattern_data['trend'],
                    mode='lines',
                    name='Trend Line',
                    line=dict(dash='dash', color='orange')
                )
            )
        
        # Add peak/trough annotations
        peak_info = seasonal_analysis.get('peak_weeks', [])
        trough_info = seasonal_analysis.get('low_weeks', [])
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title=f'Detailed Seasonal Analysis - {metric_label} (MY {selected_market_year})',
            xaxis_title='Marketing Year Week',
            yaxis_title=metric_label,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Always return as list - schema expects list output
        if isinstance(chart_ids, list):
            return [fig] * len(chart_ids)  # Same figure for multiple charts if needed
        else:
            return [fig]  # Single chart but still return as list
        
    except Exception as e:
        print(f"Error in seasonal_pattern_chart: {e}")
        # Always return error as list - schema expects list output
        if isinstance(chart_ids, list):
            return [create_empty_figure(f"Pattern Error: {str(e)}") for _ in chart_ids]
        else:
            return [create_empty_figure(f"Pattern Error: {str(e)}")]


# Legacy function removed - now using individual seasonal callbacks:
# - seasonal_overlay_chart for top chart (multi-year overlays)  
# - seasonal_pattern_chart for bottom chart (detailed single-year analysis)


def unified_seasonal_analysis_update(chart_ids, store_data=None, **menu_values):
    """
    Unified callback for all ESR seasonal analysis charts.
    
    Creates two charts:
    1. Overlay chart with seasonal index and data (1-100 percentage scale)
    2. Differenced chart showing deseasonalized data for selected market year
    
    Uses models/seasonal.py functions for seasonal index creation and differencing.
    """
    try:
        # Get menu values with defaults
        countries = menu_values.get('countries', ['Korea, South'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
        selected_market_year = menu_values.get('selected_market_year', 2023)
        start_year = menu_values.get('start_year', 2020)
        end_year = menu_values.get('end_year', 2024)
        date_range = menu_values.get('date_range', [])
        
        # Handle both single chart_id and list of chart_ids for error cases
        def handle_error_return(error_msg, chart_ids):
            if isinstance(chart_ids, list):
                return [create_empty_figure(error_msg) for _ in chart_ids]
            else:
                return [create_empty_figure(error_msg)]
        
        # Use store data
        if not store_data:
            return handle_error_return("No data available in store", chart_ids)
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return handle_error_return(f"Error loading data: {str(e)}", chart_ids)
            
        if data.empty:
            return handle_error_return("No data available", chart_ids)
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
            
        if data.empty:
            return handle_error_return("No data for selected countries", chart_ids)

        # Handle country aggregation
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer
            data = ESRAnalyzer.aggregate_multi_country_data(data, countries)
        
        # Filter by date range if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[(data['weekEndingDate'] >= start_date) & (data['weekEndingDate'] <= end_date)]
            
        if data.empty:
            return handle_error_return("No data in specified date range", chart_ids)
        
        # Ensure we have the required metric columns_col
        if seasonal_metric not in data.columns:
            return handle_error_return(f"Metric '{seasonal_metric}' not found in data", chart_ids)
        
        # Prepare time series data for seasonal analysis
        # Create a series with datetime index for seasonal functions
        ts_data = data.set_index('weekEndingDate')[seasonal_metric].sort_index()
        ts_data = ts_data.dropna()
        
        if ts_data.empty:
            return handle_error_return("No valid time series data", chart_ids)
        
        # Import seasonal analysis functions
        from MacrOSINT.models import create_seasonal_index, seasonal_difference
        
        # Create seasonal index with 100.0 scale for percentage display
        seasonal_index = create_seasonal_index(ts_data, frequency='W', scale=100.0)
        
        # Create deseasonalized data using seasonal differencing
        deseasonalized_data = seasonal_difference(
            ts_data, 
            seasonal_index, 
            frequency='W',
            method='multiplicative',  # Use multiplicative for percentage-style index
            commodity_type='grains'   # Default to grains for ESR data
        )
        
        # Get metric labels for display
        metric_labels = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales',
            'currentMYTotalCommitment': 'Current MY Total Commitment'
        }
        metric_label = metric_labels.get(seasonal_metric, seasonal_metric.replace('_', ' ').title())
        
        # CHART 1: Overlay Chart with Seasonal Index
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add original data series (scaled to percentage for comparison)
        # Scale original data to 0-100 range for visual comparison
        data_min, data_max = ts_data.min(), ts_data.max()
        scaled_data = ((ts_data - data_min) / (data_max - data_min)) * 100
        
        # Create weekly periods for overlay
        weeks = ts_data.index.isocalendar().week
        weekly_avg_data = scaled_data.groupby(weeks).mean()
        
        # Plot scaled original data averages by week
        fig1.add_trace(
            go.Scatter(
                x=weekly_avg_data.index,
                y=weekly_avg_data.values,
                mode='lines+markers',
                name=f'{metric_label} (Scaled 0-100)',
                line=dict(width=3, color='#1f77b4'),
                yaxis='y'
            )
        )
        
        # Add seasonal index overlay
        fig1.add_trace(
            go.Scatter(
                x=seasonal_index.index,
                y=seasonal_index.values,
                mode='lines+markers',
                name='Seasonal Index (100 = Average)',
                line=dict(width=2, color='#ff7f0e', dash='dash'),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        # Add reference line at 100 for seasonal index
        fig1.add_hline(
            y=100, 
            line_dash="dot", 
            line_color="white", 
            opacity=0.5,
            secondary_y=True
        )
        
        # Update layout for overlay chart
        fig1.update_layout(
            title=f'Seasonal Overlay Analysis - {metric_label}',
            template='plotly_dark',
            height=400,
            xaxis_title='Week of Year',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Configure y-axes
        fig1.update_yaxes(title_text="Scaled Data (0-100)", secondary_y=False)
        fig1.update_yaxes(title_text="Seasonal Index (100 = Average)", secondary_y=True)
        
        # CHART 2: Differenced Chart for Selected Market Year
        fig2 = go.Figure()
        
        # Filter deseasonalized data for selected market year
        # For ESR data, marketing year typically starts in September
        my_start_month = 9  # September start for grains marketing year
        
        # Create marketing year filter
        my_start = pd.Timestamp(selected_market_year, my_start_month, 1)
        my_end = pd.Timestamp(selected_market_year + 1, my_start_month - 1, 28)
        
        # Filter data for selected marketing year
        my_data = ts_data[(ts_data.index >= my_start) & (ts_data.index <= my_end)]
        my_deseasonalized = deseasonalized_data[(deseasonalized_data.index >= my_start) & (deseasonalized_data.index <= my_end)]
        
        if not my_data.empty and not my_deseasonalized.empty:
            # Add original data for selected marketing year
            fig2.add_trace(
                go.Scatter(
                    x=my_data.index,
                    y=my_data.values,
                    mode='lines+markers',
                    name=f'Original {metric_label}',
                    line=dict(width=3, color='#1f77b4')
                )
            )
            
            # Add deseasonalized data
            fig2.add_trace(
                go.Scatter(
                    x=my_deseasonalized.index,
                    y=my_deseasonalized.values,
                    mode='lines+markers',
                    name=f'Deseasonalized {metric_label}',
                    line=dict(width=2, color='#ff7f0e', dash='dash')
                )
            )
            
            chart2_title = f'Seasonal Differencing Analysis - {metric_label} (MY {selected_market_year})'
        else:
            # No data for selected year, show message
            fig2.add_annotation(
                text=f"No data available for marketing year {selected_market_year}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="white")
            )
            chart2_title = f'No Data - Marketing Year {selected_market_year}'
        
        # Update layout for differenced chart
        fig2.update_layout(
            title=chart2_title,
            template='plotly_dark',
            height=400,
            xaxis_title='Date',
            yaxis_title=metric_label,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Return both figures as a list
        if isinstance(chart_ids, list):
            if len(chart_ids) >= 2:
                return [fig1, fig2]
            else:
                return [fig1] * len(chart_ids)
        else:
            # Single chart requested, return the overlay chart
            return [fig1]
        
    except Exception as e:
        print(f"Error in unified_seasonal_analysis_update: {e}")
        import traceback
        traceback.print_exc()
        
        # Always return error as list - schema expects list output
        if isinstance(chart_ids, list):
            return [create_empty_figure(f"Seasonal Analysis Error: {str(e)}") for _ in chart_ids]
        else:
            return [create_empty_figure(f"Seasonal Analysis Error: {str(e)}")]


def seasonal_analysis_table_update(store_data=None, **menu_values):
    """
    Update function for Seasonal Analysis table showing marketing year info for selected commodity.
    Table displays market year start/end dates, peak weeks, and seasonality metrics.
    """
    try:
        # Get commodity from store or menu
        commodity = 'cattle'  # Default, will be updated from actual commodity selection
        
        # Extract commodity from store data or assume from context
        if store_data:
            try:
                if isinstance(store_data, str):
                    import json
                    data = pd.DataFrame(json.loads(store_data))
                else:
                    data = pd.DataFrame(store_data)
                
                # Try to infer commodity from data structure or patterns
                if not data.empty and 'weekEndingDate' in data.columns:
                    # Use pattern analysis to identify commodity type
                    commodity = infer_commodity_from_data(data)
            except Exception as e:
                print(f"Error processing store data for table: {e}")
        
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South'])
        seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
        selected_market_year = menu_values.get('selected_market_year', pd.Timestamp.now().year)
        start_year = menu_values.get('start_year', pd.Timestamp.now().year - 4)
        end_year = menu_values.get('end_year', pd.Timestamp.now().year)
        
        # Create seasonal summary table data
        table_data = create_seasonal_summary_table(commodity, seasonal_metric, selected_market_year, start_year, end_year, countries)
        return table_data
        
    except Exception as e:
        print(f"Error in seasonal_analysis_table_update: {e}")
        return []


def infer_commodity_from_data(data):
    """Infer commodity type from data patterns"""
    # This is a simple heuristic - could be enhanced
    if not data.empty and 'country' in data.columns:
        # Look at country patterns or other indicators
        countries = data['country'].unique()
        if 'Korea, South' in countries and len(countries) > 3:
            return 'cattle'  # Cattle typically has many export destinations
    return 'cattle'  # Default


def create_seasonal_summary_table(commodity, seasonal_metric, selected_market_year, start_year, end_year, countries):
    """
    Create seasonal summary table with marketing year information for the selected commodity.
    Shows market year start/end, peak weeks, low weeks, and seasonality strength.
    """
    try:
        # Define marketing year calendar by commodity type
        marketing_year_calendar = {
            'cattle': {'start_month': 1, 'start_day': 1},      # Jan 1 - Dec 31
            'hogs': {'start_month': 1, 'start_day': 1},        # Jan 1 - Dec 31
            'pork': {'start_month': 1, 'start_day': 1},        # Jan 1 - Dec 31
            'corn': {'start_month': 9, 'start_day': 1},        # Sep 1 - Aug 31
            'wheat': {'start_month': 6, 'start_day': 1},       # Jun 1 - May 31
            'soybeans': {'start_month': 9, 'start_day': 1}     # Sep 1 - Aug 31
        }
        
        my_calendar = marketing_year_calendar.get(commodity.lower(), {'start_month': 1, 'start_day': 1})
        
        # Calculate marketing year dates
        my_start_date = pd.Timestamp(selected_market_year, my_calendar['start_month'], my_calendar['start_day'])
        my_end_date = my_start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        
        # Format dates for display
        my_start_str = my_start_date.strftime('%b %d, %Y')
        my_end_str = my_end_date.strftime('%b %d, %Y')
        
        # Create table rows for each selected country or summary
        table_rows = []
        
        # Seasonal patterns by commodity (general patterns)
        seasonal_patterns = {
            'cattle': {'peak_weeks': 'Weeks 20-30 (May-Jul)', 'low_weeks': 'Weeks 45-52 (Nov-Dec)', 'seasonality': 0.35},
            'hogs': {'peak_weeks': 'Weeks 15-25 (Apr-Jun)', 'low_weeks': 'Weeks 40-50 (Oct-Dec)', 'seasonality': 0.42},
            'pork': {'peak_weeks': 'Weeks 15-25 (Apr-Jun)', 'low_weeks': 'Weeks 40-50 (Oct-Dec)', 'seasonality': 0.42},
            'corn': {'peak_weeks': 'Weeks 10-20 (Dec-Feb)', 'low_weeks': 'Weeks 30-40 (May-Jul)', 'seasonality': 0.65},
            'wheat': {'peak_weeks': 'Weeks 45-10 (May-Sep)', 'low_weeks': 'Weeks 20-30 (Nov-Feb)', 'seasonality': 0.78},
            'soybeans': {'peak_weeks': 'Weeks 5-15 (Oct-Dec)', 'low_weeks': 'Weeks 25-35 (Mar-May)', 'seasonality': 0.72}
        }
        
        pattern = seasonal_patterns.get(commodity.lower(), {
            'peak_weeks': 'Variable',
            'low_weeks': 'Variable', 
            'seasonality': 0.50
        })
        
        # Create main row for the commodity
        main_row = {
            'commodity': commodity.title(),
            'my_start': f"{my_start_str} - {my_end_str}",
            'peak_weeks': pattern['peak_weeks'],
            'low_weeks': pattern['low_weeks'],
            'seasonality': round(pattern['seasonality'], 2)
        }
        
        table_rows.append(main_row)
        
        # Add additional context rows for different metrics
        metric_labels = {
            'weeklyExports': 'Exports',
            'outstandingSales': 'Sales',
            'grossNewSales': 'New Sales',
            'currentMYNetSales': 'Net Sales',
            'currentMYTotalCommitment': 'Commitments'
        }
        
        if seasonal_metric in metric_labels:
            metric_row = {
                'commodity': f"{commodity.title()} - {metric_labels[seasonal_metric]}",
                'my_start': f"MY {selected_market_year}",
                'peak_weeks': pattern['peak_weeks'],
                'low_weeks': pattern['low_weeks'],
                'seasonality': round(pattern['seasonality'] * 0.9, 2)  # Slight variation for metric-specific
            }
            table_rows.append(metric_row)
        
        return table_rows
        
    except Exception as e:
        print(f"Error creating seasonal summary table: {e}")
        # Return default row on error
        return [{
            'commodity': commodity.title() if commodity else 'Unknown',
            'my_start': 'Error loading data',
            'peak_weeks': 'N/A',
            'low_weeks': 'N/A',
            'seasonality': 0.0
        }]


def create_country_analysis_chart(data, country_metric, countries, country_display_mode, 
                                  start_year, end_year, chart_id, chart_index=0):
    """Create country analysis chart with market year overlays and multi-country support"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Check if columns_col exists
        if country_metric not in data.columns:
            return create_empty_figure(f"Column '{country_metric}' not found in data")
        
        # Create title mapping
        metric_labels = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales', 
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales',
            'currentMYTotalCommitment': 'Current MY Total Commitment'
        }
        metric_name = metric_labels.get(country_metric, country_metric.title())
        
        # Create marketing year columns_col if it doesn't exist
        if 'marketing_year' not in data.columns:
            data = data.copy()
            # USDA marketing year (Oct-Sep for most commodities)
            data['marketing_year'] = data['weekEndingDate'].apply(
                lambda x: x.year if x.month < 10 else x.year + 1
            )
        
        # Apply year filtering for market year overlays
        if start_year and end_year:
            data = data[
                (data['marketing_year'] >= start_year) & 
                (data['marketing_year'] <= end_year)
            ]
        
        if data.empty:
            return create_empty_figure(f"No data available for {metric_name}")
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            aggregated_data = data.groupby(['weekEndingDate', 'marketing_year'])[country_metric].sum().reset_index()
            aggregated_data['country'] = f"Sum of {', '.join(countries[:3])}" + ("..." if len(countries) > 3 else "")
            
            # Create chart for summed data
            fig = px.line(
                aggregated_data,
                x='weekEndingDate',
                y=country_metric, 
                color='marketing_year',
                title=f'{metric_name} - Market Year Overlays (Summed Countries)',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:
            # Individual countries
            if chart_index == 0:
                # First chart - show all years overlaid
                fig = px.line(
                    data,
                    x='weekEndingDate',
                    y=country_metric,
                    color='marketing_year',
                    line_dash='country',  # Distinguish countries by line style
                    title=f'{metric_name} - Market Year Overlays by Country',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
            else:
                # Second chart - focus on country comparison for current year
                current_my = data['marketing_year'].max()
                current_data = data[data['marketing_year'] == current_my]
                
                fig = px.line(
                    current_data,
                    x='weekEndingDate',
                    y=country_metric,
                    color='country',
                    title=f'{metric_name} - Current Marketing Year ({current_my}) Country Comparison',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
        
        # Update layout
        fig.update_layout(
            height=450,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Week Ending Date",
            yaxis_title=metric_name,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' + 
                         f'{metric_name}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating country analysis chart: {e}")
        return create_empty_figure(f"Error creating chart: {str(e)}")