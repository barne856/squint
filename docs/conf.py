
extensions = [ "breathe", "sphinx.ext.mathjax", "sphinxcontrib.tikz" ]
project = "SQUINT"
breathe_default_project = "SQUINT"
tikz_latex_preamble = r'''
\usepackage{tikz}
\usetikzlibrary{matrix,calc}
'''
html_theme = 'furo'

# Customize the theme
html_theme_options = {
    'logo_only': True,
    'sidebar_hide_name': True,
    'light_logo': 'logo-light.png',
    'dark_logo': 'logo-dark.png',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    "dark_css_variables": {
        "color-foreground-primary": "#F0F0F0",
        "color-foreground-secondary": "#B0B0B0",
        "color-background-primary": "#0A0A0A",
        "color-background-secondary": "#1A1A1A",
        "color-brand-primary": "#DBDBDB",
        "color-brand-content": "#00BFFF",
        "color-admonition-background": "#005A82"
    }
}

html_static_path = ['_static']

# custom css
def setup(app):
    app.add_css_file('custom.css')
