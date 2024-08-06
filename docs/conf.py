extensions = [ "breathe" ]
breathe_default_project = "SQUINT"


# Option 1: Read the Docs theme
html_theme = 'sphinx_rtd_theme'
# Option 2: Furo theme
# html_theme = 'furo'
# Option 3: Material theme
# html_theme = 'sphinx_material'
# Option 4: Book theme
# html_theme = 'sphinx_book_theme'
# Customize the theme
html_theme_options = {
    'logo_only': False,
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
    'titles_only': False
}