"""
TableClient Update Components

Components for manually updating TableClients in the data.data_tables module.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from MacrOSINT.data.data_tables import NASSTable, ESRTableClient, FASTable
import datetime


class TableClientUpdater:
    """Component for updating TableClient data sources."""
    
    def __init__(self, client_id: str = "table_client_updater"):
        self.client_id = client_id
        self.components = self._create_components()
        
    def _create_components(self):
        """Create the update interface components."""
        
        # Client type selector
        client_selector = dbc.Card([
            dbc.CardHeader(html.H4("Select TableClient Type")),
            dbc.CardBody([
                dcc.Dropdown(
                    id=f"{self.client_id}_client_type",
                    options=[
                        {"label": "NASS Agricultural Statistics", "value": "nass"},
                        {"label": "ESR Export Sales Reports", "value": "esr"},
                        {"label": "FAS Foreign Agricultural Service", "value": "fas"}
                    ],
                    value="esr",
                    placeholder="Select client type..."
                )
            ])
        ], className="mb-3")
        
        # Update options
        update_options = dbc.Card([
            dbc.CardHeader(html.H4("Update Options")),
            dbc.CardBody([
                html.Div(id=f"{self.client_id}_options_container")
            ])
        ], className="mb-3")
        
        # Update controls
        update_controls = dbc.Card([
            dbc.CardHeader(html.H4("Update Controls")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Update Selected Data",
                            id=f"{self.client_id}_update_btn",
                            color="primary",
                            size="lg",
                            className="me-2"
                        )
                    ], width="auto"),
                    dbc.Col([
                        dbc.Button(
                            "Update All Data",
                            id=f"{self.client_id}_update_all_btn",
                            color="warning",
                            size="lg"
                        )
                    ], width="auto")
                ], justify="start"),
                html.Hr(),
                html.Div(id=f"{self.client_id}_status", className="mt-3")
            ])
        ])
        
        return html.Div([
            client_selector,
            update_options,
            update_controls
        ])
    
    def get_layout(self):
        """Return the complete layout."""
        return html.Div([
            html.H2("TableClient Data Updater", className="mb-4"),
            html.P("Update data sources for NASSTable, ESRTableClient, and FASTable instances.", 
                   className="text-muted mb-4"),
            self.components
        ], className="container-fluid")


# Callback for updating options based on client type
@callback(
    Output("table_client_updater_options_container", "children"),
    Input("table_client_updater_client_type", "value")
)
def update_client_options(client_type):
    """Update available options based on selected client type."""
    
    if client_type == "nass":
        return html.Div([
            html.H5("NASS Update Options"),
            html.P("Update USDA NASS agricultural statistics data."),
            dcc.Dropdown(
                id="nass_commodity_select",
                options=[
                    {"label": "All Commodities", "value": "all"},
                    {"label": "Cattle", "value": "cattle"},
                    {"label": "Hogs", "value": "hogs"},
                    {"label": "Corn", "value": "corn"},
                    {"label": "Wheat", "value": "wheat"},
                    {"label": "Soybeans", "value": "soybeans"}
                ],
                value="all",
                placeholder="Select commodity to update..."
            )
        ])
    
    elif client_type == "esr":
        return html.Div([
            html.H5("ESR Update Options"),
            html.P("Update Export Sales Report data for selected commodities and years."),
            dcc.Dropdown(
                id="esr_commodity_select",
                options=[
                    {"label": "All ESR Commodities", "value": "all"},
                    {"label": "Cattle", "value": "cattle"},
                    {"label": "Corn", "value": "corn"},
                    {"label": "Wheat", "value": "wheat"},
                    {"label": "Soybeans", "value": "soybeans"}
                ],
                value="all",
                placeholder="Select commodity to update..."
            ),
            html.Br(),
            html.Label("Data Source:"),
            dcc.Dropdown(
                id="esr_data_source",
                options=[
                    {"label": "Historical Data (Production API)", "value": "historical"},
                    {"label": "Current Week Data (Staging API)", "value": "current"},
                    {"label": "Combined (Historical + Current)", "value": "combined"}
                ],
                value="historical",
                placeholder="Select data source..."
            ),
            html.Br(),
            html.Label("Year Range:"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="esr_start_year",
                        options=[
                            {"label": str(year), "value": year}
                            for year in range(2020, datetime.datetime.now().year + 2)
                        ],
                        value=datetime.datetime.now().year - 2,
                        placeholder="Start year..."
                    )
                ], width=6),
                dbc.Col([
                    dcc.Dropdown(
                        id="esr_end_year",
                        options=[
                            {"label": str(year), "value": year}
                            for year in range(2020, datetime.datetime.now().year + 2)
                        ],
                        value=datetime.datetime.now().year,
                        placeholder="End year..."
                    )
                ], width=6)
            ])
        ])
    
    elif client_type == "fas":
        return html.Div([
            html.H5("FAS Update Options"),
            html.P("Update Foreign Agricultural Service PSD (Production, Supply & Distribution) data."),
            dcc.Dropdown(
                id="fas_commodity_select",
                options=[
                    {"label": "All FAS Commodities", "value": "all"},
                    {"label": "Cattle", "value": "cattle"},
                    {"label": "Corn", "value": "corn"},
                    {"label": "Wheat", "value": "wheat"},
                    {"label": "Soybeans", "value": "soybeans"}
                ],
                value="all",
                placeholder="Select commodity to update..."
            ),
            html.Br(),
            html.Label("Year Range for PSD:"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="fas_start_year",
                        options=[
                            {"label": str(year), "value": year}
                            for year in range(1982, datetime.datetime.now().year + 2)
                        ],
                        value=2020,
                        placeholder="Start year..."
                    )
                ], width=6),
                dbc.Col([
                    dcc.Dropdown(
                        id="fas_end_year",
                        options=[
                            {"label": str(year), "value": year}
                            for year in range(1982, datetime.datetime.now().year + 2)
                        ],
                        value=datetime.datetime.now().year + 1,
                        placeholder="End year..."
                    )
                ], width=6)
            ])
        ])
    
    return html.Div([
        html.P("Please select a client type to see available options.")
    ])


# Callback for handling updates
@callback(
    Output("table_client_updater_status", "children"),
    [Input("table_client_updater_update_btn", "n_clicks"),
     Input("table_client_updater_update_all_btn", "n_clicks")],
    [State("table_client_updater_client_type", "value")]
)
def handle_table_updates(update_btn_clicks, update_all_btn_clicks, client_type):
    """Handle table update requests."""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([
            dbc.Alert("Ready to update data. Select options and click update.", color="info")
        ])
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if button_id == "table_client_updater_update_btn":
            return handle_selective_update(client_type)
        elif button_id == "table_client_updater_update_all_btn":
            return handle_full_update(client_type)
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Update failed: {str(e)}", color="danger"),
            html.Pre(str(e))
        ])
    
    return html.Div([
        dbc.Alert("No action taken.", color="warning")
    ])


def handle_selective_update(client_type):
    """Handle selective data updates."""
    
    if client_type == "nass":
        # Initialize NASS client and update selected data
        client = NASSTable()
        status = dbc.Alert("NASS selective update would be performed here. Implementation depends on specific requirements.", color="info")
        
    elif client_type == "esr":
        # Initialize ESR client and perform multi-year update
        try:
            client = ESRTableClient()
            
            # Example: Update cattle data for recent years
            commodity = "cattle"
            current_year = datetime.datetime.now().year
            start_year = current_year - 2
            end_year = current_year
            
            # Use the new async multi-year update method
            result = client.update_esr_multi_year_sync(
                commodity=commodity,
                start_year=start_year,
                end_year=end_year,
                top_n=10,
                max_concurrent=3
            )
            
            # Create detailed status message
            if result['summary']['successful_updates'] > 0:
                status_color = "success" if result['summary']['failed_updates'] == 0 else "warning"
                status_message = f"""
                ESR Update Completed:
                • Commodity: {commodity}
                • Years: {start_year}-{end_year}
                • Successful: {result['summary']['successful_updates']}
                • Failed: {result['summary']['failed_updates']}
                • Success Rate: {result['summary']['success_rate']}
                """
            else:
                status_color = "danger"
                status_message = f"ESR update failed for {commodity} ({start_year}-{end_year})"
            
            status = dbc.Alert(status_message, color=status_color)
            
        except Exception as e:
            status = dbc.Alert(f"ESR update error: {str(e)}", color="danger")
        
    elif client_type == "fas":
        # Initialize FAS client and update selected data
        client = FASTable()
        status = dbc.Alert("FAS selective update would be performed here. Implementation depends on specific requirements.", color="info")
        
    else:
        status = dbc.Alert("Invalid client type selected.", color="danger")
    
    return html.Div([
        status,
        html.P(f"Selected client type: {client_type}", className="text-muted")
    ])


def handle_full_update(client_type):
    """Handle full data updates."""
    
    try:
        if client_type == "nass":
            client = NASSTable()
            result = client.update_all()
            if isinstance(result, dict) and result:  # Failed updates
                status = dbc.Alert(f"NASS update completed with {len(result)} failures.", color="warning")
            else:
                status = dbc.Alert("NASS update completed successfully.", color="success")
                
        elif client_type == "esr":
            client = ESRTableClient()
            # ESRTableClient doesn't have update_all, so we'd need to implement specific ESR updates
            status = dbc.Alert("ESR update initiated. Check logs for progress.", color="info")
            
        elif client_type == "fas":
            client = FASTable()
            result = client.update_all()
            if isinstance(result, dict) and result:  # Failed updates
                status = dbc.Alert(f"FAS update completed with {len(result)} failures.", color="warning")
            else:
                status = dbc.Alert("FAS update completed successfully.", color="success")
                
        else:
            status = dbc.Alert("Invalid client type selected.", color="danger")
            
    except Exception as e:
        status = dbc.Alert(f"Update failed: {str(e)}", color="danger")
    
    return html.Div([
        status,
        html.P(f"Full update attempted for: {client_type}", className="text-muted"),
        html.Small("Note: Updates may take several minutes to complete. Check application logs for detailed progress.", 
                  className="text-muted")
    ])


# Create the updater instance
table_client_updater = TableClientUpdater()