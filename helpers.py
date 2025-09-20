import streamlit as st

def check_authentication():
    """
    Checks if the user is authenticated by verifying session state.
    If not authenticated, it redirects the user to the login page.
    This function should be called at the beginning of every restricted page.
    """
    if not st.session_state.get("authenticated"):
        # If the user is not authenticated, redirect to the login page
        st.switch_page("login.py")
    return st.session_state.get("role")

def render_sidebar():
    """
    Renders a dynamic sidebar with pages visible based on the user's role.
    This uses custom CSS to hide navigation links for unauthorized pages, which is
    the most reliable method in Streamlit for controlling page visibility.
    """
    # Ensure the user is authenticated before showing the sidebar
    role = check_authentication()

    # Define which pages are visible to which roles.
    # The key is the page title as it appears in the sidebar (from the filename).
    # The value is a list of roles that can view that page.
    page_permissions = {
        "Dataset_Upload_&Preprocessing": ["admin"],
        "Admin_Dashboard": ["admin"],
        "Model_Selection&_Training": ["admin"],
        "Inference": ["admin", "user"]  # Both roles can see the Inference page
    }

    # Use CSS to selectively hide sidebar links.
    # This is more robust than trying to conditionally render the links.
    st.markdown(
        """
        <style>
            /* Hide the Streamlit native menu and footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* Custom styling for the sidebar navigation */
            [data-testid="st-sidebar-nav"] ul {
                list-style-type: none;
                padding-left: 0;
            }
            [data-testid="st-sidebar-nav"] ul li {
                display: block; /* Ensure list items are block elements */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Iterate through the page permissions and hide links if the user's role doesn't match.
    for page, allowed_roles in page_permissions.items():
        if role not in allowed_roles:
            # This CSS selector targets the sidebar navigation item (li) that contains a link (a)
            # whose 'href' attribute contains the page name. This is how we hide specific pages.
            st.markdown(
                f"""
                <style>
                    li[data-testid*="st-sidebar-nav-item"] a[href*="{page}"] {{
                        display: none;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
