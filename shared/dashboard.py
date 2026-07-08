"""Dashboard and Report Builder helper functions for low-code reports."""

import streamlit as st
import time
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def init_dashboard_state(page_id: str, default_widgets: list[str]):
    """Initialize active and collapsed widgets in session state from URL query parameters if present."""
    if f"{page_id}_active_widgets" not in st.session_state:
        # Check query params
        query = st.query_params
        if "widgets" in query:
            widgets_str = query["widgets"]
            active = [w.strip() for w in widgets_str.split(",") if w.strip()]
        else:
            active = list(default_widgets)
        st.session_state[f"{page_id}_active_widgets"] = active

    if f"{page_id}_collapsed_widgets" not in st.session_state:
        query = st.query_params
        if "collapsed" in query:
            collapsed_str = query["collapsed"]
            collapsed = set([w.strip() for w in collapsed_str.split(",") if w.strip()])
        else:
            collapsed = set()
        st.session_state[f"{page_id}_collapsed_widgets"] = collapsed

def render_widget_controls(page_id: str, index: int, widget_id: str, widget_name: str) -> bool:
    """Render the dashboard control header for a widget. Returns True if the widget is collapsed."""
    active = st.session_state[f"{page_id}_active_widgets"]
    collapsed = st.session_state[f"{page_id}_collapsed_widgets"]
    is_collapsed = widget_id in collapsed

    # Create a nice control bar with styling
    col_title, col_up, col_down, col_collapse, col_remove = st.columns([12, 1, 1, 1, 1])
    
    with col_title:
        st.markdown(f"#### {widget_name}")
        
    # Up button
    if col_up.button("▲", key=f"{page_id}_{widget_id}_up", help="Move Up", use_container_width=True):
        if index > 0:
            active[index], active[index-1] = active[index-1], active[index]
            st.session_state[f"{page_id}_active_widgets"] = active
            st.rerun()
            
    # Down button
    if col_down.button("▼", key=f"{page_id}_{widget_id}_down", help="Move Down", use_container_width=True):
        if index < len(active) - 1:
            active[index], active[index+1] = active[index+1], active[index]
            st.session_state[f"{page_id}_active_widgets"] = active
            st.rerun()
            
    # Collapse/Expand button
    if col_collapse.button("➖" if not is_collapsed else "➕", key=f"{page_id}_{widget_id}_coll", help="Collapse/Expand", use_container_width=True):
        if is_collapsed:
            collapsed.remove(widget_id)
        else:
            collapsed.add(widget_id)
        st.session_state[f"{page_id}_collapsed_widgets"] = collapsed
        st.rerun()
        
    # Remove button
    if col_remove.button("❌", key=f"{page_id}_{widget_id}_rem", help="Remove Widget", use_container_width=True):
        if widget_id in active:
            active.remove(widget_id)
        st.session_state[f"{page_id}_active_widgets"] = active
        st.rerun()
        
    if is_collapsed:
        st.info("Widget collapsed. Click ➕ to expand.")
        
    return is_collapsed

def render_dashboard_header(page_id: str, all_widgets_dict: dict[str, str], config_title: str = "🛠️ Low-Code Report Configuration", data_dict: dict = None):
    """Render report configuration panel containing add widgets, share web app, download PDF, and send email actions."""
    init_dashboard_state(page_id, list(all_widgets_dict.keys()))
    active = st.session_state[f"{page_id}_active_widgets"]
    
    with st.expander(config_title, expanded=False):
        # 1. Widget Selection Box (to add back removed widgets)
        available = [w for w in all_widgets_dict.keys() if w not in active]
        
        col_add, col_actions = st.columns([1, 1])
        
        with col_add:
            st.markdown("**Add Widgets**")
            if available:
                selected_label = st.selectbox(
                    "Choose a widget to add to your report:",
                    options=["Select widget..."] + [all_widgets_dict[w] for w in available],
                    key=f"{page_id}_add_widget_select"
                )
                if selected_label != "Select widget...":
                    # Find matching widget ID
                    selected_id = [k for k, v in all_widgets_dict.items() if v == selected_label][0]
                    active.append(selected_id)
                    st.session_state[f"{page_id}_active_widgets"] = active
                    st.rerun()
            else:
                st.info("All widgets are currently active in the report layout.")
                
        with col_actions:
            st.markdown("**Report Sharing Actions**")
            
            # Serialize state
            widgets_val = ",".join(active)
            collapsed_val = ",".join(st.session_state[f"{page_id}_collapsed_widgets"])
            
            # 2. Share Web App
            if st.button("🔗 Get Shareable Web App Link", key=f"{page_id}_share_btn", use_container_width=True):
                # Update query params in-place
                st.query_params["widgets"] = widgets_val
                st.query_params["collapsed"] = collapsed_val
                
                st.success("Report configuration saved to query parameters!")
                # Show copyable URL
                share_url = f"?widgets={widgets_val}&collapsed={collapsed_val}"
                st.code(share_url, language="text")
                st.caption("Copy the query string suffix above and attach it to your browser URL to share this layout.")

            # 3. Export PDF
            if data_dict:
                pdf_data = generate_pdf_bytes(page_id, active, data_dict)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"{page_id}_custom_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"{page_id}_download_pdf_btn"
                )
                
                # 4. Email simulator
                with st.popover("✉️ Send Report via Email", use_container_width=True):
                    email_to = st.text_input("Recipient Email", placeholder="operator@company.com", key=f"{page_id}_email_to")
                    email_subject = st.text_input("Subject", value=f"Custom Report - {page_id.replace('_', ' ').title()}", key=f"{page_id}_email_sub")
                    email_body = st.text_area("Message", value=f"Hi,\n\nPlease find attached the custom report for the {page_id.replace('_', ' ').title()} dashboard layout.\n\nBest regards,", key=f"{page_id}_email_body")
                    
                    if st.button("Send Email", key=f"{page_id}_email_send_btn", use_container_width=True):
                        if not email_to:
                            st.error("Please enter a recipient email address.")
                        else:
                            with st.spinner("Generating PDF report..."):
                                time.sleep(0.6)
                            with st.spinner("Attaching report to email..."):
                                time.sleep(0.4)
                            with st.spinner("Sending email..."):
                                time.sleep(0.5)
                            st.success(f"Success! Email sent to {email_to} with {page_id}_custom_report.pdf attached.")

def generate_pdf_bytes(page_id: str, active_widgets: list[str], data_dict: dict) -> bytes:
    """Generate a clean PDF report of the active widgets using Matplotlib."""
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # PAGE 1: Cover Page / Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Header banner
        fig.patch.set_facecolor('#F5F3E7')
        ax.text(0.5, 0.88, f"WOOD WIDE AI REPORT", fontsize=24, weight='bold', color='#0B3D2E', ha='center')
        ax.text(0.5, 0.82, f"{page_id.replace('_', ' ').upper()} WORKFLOW ANALYSIS", fontsize=16, weight='bold', color='#668575', ha='center')
        
        # Meta info box
        ax.text(0.1, 0.72, f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11, color='#11231a')
        ax.text(0.1, 0.69, f"Active Widgets: {len(active_widgets)}", fontsize=11, color='#11231a')
        
        # Line break
        ax.plot([0.1, 0.9], [0.65, 0.65], color='#0B3D2E', lw=2)
        
        # Summary metrics
        ax.text(0.1, 0.58, "SUMMARY METRICS", fontsize=14, weight='bold', color='#0B3D2E')
        y_pos = 0.52
        
        metrics = data_dict.get("metrics", {})
        if not metrics:
            metrics = {
                "Total Scored": data_dict.get("total_scored", "N/A"),
                "Threshold": data_dict.get("threshold", "N/A"),
                "Flagged/Risky Cohort Size": data_dict.get("flagged_size", "N/A")
            }
            
        for label, val in metrics.items():
            ax.text(0.12, y_pos, f"• {label}: {val}", fontsize=12, color='#11231a')
            y_pos -= 0.045
            
        # Add a placeholder statement about low-code layout customization
        ax.text(0.1, 0.15, "Low-Code Configuration Note:", fontsize=10, weight='bold', color='#668575')
        ax.text(0.1, 0.12, f"This report was customized dynamically. Order of widgets: {', '.join(active_widgets)}.", fontsize=9, color='#668575')
        
        pdf.savefig(fig)
        plt.close()
        
        # PAGE 2: Visualizations Page
        # Gather all pie charts or distributions if active
        pie_charts = [w for w in active_widgets if "pie" in w]
        if pie_charts:
            fig, axs = plt.subplots(len(pie_charts), 1, figsize=(8.5, 11))
            fig.patch.set_facecolor('#ffffff')
            if len(pie_charts) == 1:
                axs = [axs]
                
            for idx, pie_id in enumerate(pie_charts):
                ax = axs[idx]
                pie_data = data_dict.get(f"{pie_id}_data")
                if pie_data:
                    labels = list(pie_data.keys())
                    sizes = list(pie_data.values())
                    colors = ['#0B3D2E', '#175cd3', '#f04438', '#f79009', '#668575', '#d8e2dc'][:len(labels)]
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, 
                           textprops={'fontsize': 10, 'color': '#11231a'})
                    ax.axis('equal')
                    title_label = pie_id.replace('_', ' ').title()
                    ax.set_title(title_label, fontsize=14, weight='bold', color='#0B3D2E', pad=15)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, f"No data available for {pie_id}", ha='center')
                    
            pdf.savefig(fig)
            plt.close()
            
        # PAGE 3: Key Data Preview Page (if active)
        table_widgets = [w for w in active_widgets if "plan" in w or "cohort" in w or "segmented" in w or "raw_output" in w]
        if table_widgets:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            ax.text(0.1, 0.92, "REPORT DATA EXPORTS", fontsize=16, weight='bold', color='#0B3D2E')
            
            y_tbl = 0.84
            for tbl_id in table_widgets[:2]: # Show previews of up to 2 tables
                tbl_data = data_dict.get(f"{tbl_id}_df")
                ax.text(0.1, y_tbl, f"Widget Preview: {tbl_id.replace('_', ' ').title()}", fontsize=12, weight='bold', color='#668575')
                y_tbl -= 0.04
                
                if tbl_data is not None and not tbl_data.empty:
                    # Render a clean text-based table preview (first 5 columns and 10 rows)
                    col_preview = tbl_data.columns[:5].tolist()
                    rows_preview = tbl_data[col_preview].head(10)
                    
                    # Construct text-based table
                    lines = []
                    header_line = " | ".join([f"{col[:12]}" for col in col_preview])
                    lines.append(header_line)
                    lines.append("-" * len(header_line))
                    for _, row in rows_preview.iterrows():
                        row_line = " | ".join([f"{str(val)[:12]}" for val in row.values])
                        lines.append(row_line)
                        
                    table_text = "\n".join(lines)
                    ax.text(0.12, y_tbl, table_text, fontfamily='monospace', fontsize=8, color='#11231a', va='top')
                    y_tbl -= (len(lines) * 0.02 + 0.06)
                else:
                    ax.text(0.12, y_tbl, "No table data available", fontsize=10, style='italic', color='#668575')
                    y_tbl -= 0.08
                    
            pdf.savefig(fig)
            plt.close()

    return pdf_buffer.getvalue()
