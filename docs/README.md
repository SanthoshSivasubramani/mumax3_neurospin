# MuMax3-SAF-NeuroSpin Documentation Site

This directory contains the GitHub Pages documentation for MuMax3-SAF-NeuroSpin v2.1.0.

## Contents

- `_config.yml` - Jekyll configuration
- `_layouts/` - Page templates
- `index.md` - Homepage
- `api/` - API reference documentation
- `examples/` - Example scripts
- `tutorials/` - Step-by-step guides
- `releases/` - Download links and changelog

## Local Testing

To test the site locally:

```bash
# Install Jekyll
gem install jekyll bundler

# Create Gemfile
cat > Gemfile << EOF
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
gem "jekyll-theme-minimal"
EOF

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Visit: http://localhost:4000/mumax3_neurospin/
```

## Deployment

The site is automatically deployed via GitHub Pages when pushed to the main branch.

## Structure

```
site/
├── _config.yml           # Jekyll configuration
├── _layouts/
│   └── default.html      # Page template
├── index.md              # Homepage
├── api/
│   ├── index.md         # API overview
│   ├── v1.md            # V1.0 API docs
│   ├── v2.md            # V2.0 API docs
│   └── v21.md           # V2.1 API docs
├── examples/
│   └── index.md         # Example scripts
├── tutorials/
│   └── index.md         # Step-by-step guides
└── releases/
    └── index.md         # Downloads & changelog
```

## Updating

To update the documentation:

1. Edit the relevant `.md` files
2. Test locally with Jekyll
3. Commit and push to main branch
4. GitHub Pages will automatically rebuild

## Author

Dr. Santhosh Sivasubramani  
IIT Delhi & University of Edinburgh  
ragansanthosh@ieee.org

## Version

Site Version: v2.1.0  
Last Updated: February 3, 2026
