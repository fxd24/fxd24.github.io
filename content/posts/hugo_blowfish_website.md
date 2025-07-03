---
title: "From Zero to Live: Publishing Your Hugo Website with Blowfish and GitHub Pages"
date: 2025-07-03
draft: false
featureimage: "https://upload.wikimedia.org/wikipedia/commons/a/af/Logo_of_Hugo_the_static_website_generator.svg"
summary: "A step-by-step guide to building and deploying a Hugo website using the Blowfish theme and GitHub Pages, including custom domain setup."
heroStyle: "big"
tags: ["hugo", "blowfish", "github-pages", "web-development", "static-site"]
categories: ["tutorials", "web-development"]
---

# From Zero to Live: Publishing Your Hugo Website with Blowfish and GitHub Pages

Building and deploying a modern static website doesn't have to be complicated. In this comprehensive guide, I'll walk you through the entire process of creating a Hugo website using the Blowfish theme and publishing it to GitHub Pages with a custom domain. This is exactly the setup I used for my own website, including the gotchas and solutions I discovered along the way.

## Introduction

Static site generators have revolutionized how we build websites. Hugo, being one of the fastest and most flexible options, paired with the beautiful Blowfish theme, creates an excellent foundation for personal websites, blogs, and portfolios. Combined with GitHub Pages for hosting, you get a completely free, fast, and reliable web presence.

What you'll learn in this guide:
- Setting up Hugo and Blowfish theme using the new CLI tool
- Understanding Hugo's content structure and configuration
- Configuring menus and navigation
- Deploying to GitHub Pages with custom domains
- Troubleshooting common deployment issues

## Requirements

Before we start, you'll need to install several tools. Since I'm using macOS with Homebrew, I'll provide those instructions, but similar packages are available for other platforms.

### Essential Tools

**1. Install Go**
```bash
brew install go
```
Verify installation:
```bash
go version
```

**2. Install Hugo**
```bash
brew install hugo
```
Make sure you have Hugo 0.87.0 or later:
```bash
hugo version
```

**3. Install Git**
```bash
brew install git
```

**4. Install Node.js (for Blowfish CLI)**
```bash
brew install node
```

**5. Install Blowfish CLI Tool**
```bash
npm install -g blowfish-tools
```

### Optional but Recommended
- A GitHub account for hosting
- A custom domain (optional, but we'll cover setup)

## Setting Up Your Hugo Site with Blowfish

### The Empty Folder Requirement

Here's an important detail I discovered: **The Blowfish CLI tool requires an empty folder to work properly.** If you already have a GitHub repository set up with README files, you'll need to start fresh or work around this limitation.

I've actually created an issue with the Blowfish team to improve this, but for now, the best approach is to start with a completely empty directory.

### Creating Your Site

**1. Create and navigate to an empty directory:**
```bash
mkdir fxd24.github.io
cd fxd24.github.io
```
In your case, replace `fxd24.github.io` with your desired repository name. This will be the root of your Hugo site.

**2. Run the Blowfish CLI tool:**
```bash
blowfish-tools new
```

The CLI will guide you through an interactive setup process, asking about:
- Site name and description
- Author information
- Color scheme preferences
- Features you want to enable

**3. If you need to connect to an existing GitHub repository:**
```bash
git remote add origin https://github.com/fxd24/fxd24.github.io.git
```
Make sure to replace the URL with your own repository link.

## Understanding Hugo Content Structure

Hugo organizes content in a specific way that's important to understand:

### The Content Directory

All your website content goes in the `content/` folder. Hugo is flexible with structure - you can use simple files or bundle directories depending on your needs:

```
content/
├── about.md              # Simple page
├── posts/
│   ├── hugo-website-publishing/
│   │   ├── index.md
│   │   └── featured.png  # Thumbnail image
│   ├── simple-post.md    # Simple post without folder
│   └── another-post/
│       ├── index.md
│       └── featured.jpg
└── _index.md             # Homepage content
```

### Adding Content

**1. Create an About page (simple file):**
```bash
hugo new about.md
```

**2. Create blog posts (two approaches):**
```bash
# Simple post (single file)
hugo new posts/my-simple-post.md

# Post bundle (with images and resources)
hugo new posts/my-complex-post/index.md
```
Note: You can also create files directly using any code editor of your preference.

**3. Add featured images:**
To add a thumbnail image to any post or page:
- For single files: place `featured.png` (or `.jpg`) in the same directory as your content file
- For bundles: place `featured.png` in the same directory as your `index.md`

### Front Matter Configuration

Each content file starts with front matter that controls how the page behaves:

```yaml
---
title: "Your Page Title"
date: 2025-07-03
draft: false
tags: ["hugo", "web-development"]
categories: ["tutorials"]
---
```

Key fields:
- `title`: The page title
- `date`: Publication date
- `draft`: Set to `true` to hide from published site
- `tags` and `categories`: For organization and filtering

## Configuration

Hugo uses configuration files to control your site's behavior. Blowfish uses multiple configuration files for organization:

### Main Configuration Files

```
config/_default/
├── hugo.toml          # Main Hugo settings
├── languages.en.toml  # Language-specific settings
├── markup.toml        # Markdown processing settings
├── menus.en.toml      # Navigation menu configuration
└── params.toml        # Theme-specific parameters
```
Important: When using the Blowfish CLI there will be also a hugo.toml file in the root directory. That file needs to be deleted.

### Key Configuration Areas

**Site Information (`hugo.toml`):**
```toml
baseURL = "https://grafdavid.com" # Replace with your domain
languageCode = "en"
title = "David's Universe" # Replace with your site title
theme = "blowfish"
```

**Theme Parameters (`params.toml`):**
```toml
# Site appearance
colorScheme = "auto"
defaultAppearance = "dark"

# Homepage layout
homepage.layout = "profile"

# Social links
[author]
name = "Your Name"
headline = "Your Headline"
bio = "Your bio description"
```

## Creating Navigation Menus

Navigation is configured in `menus.en.toml`. Here's how to set up your menu items:

```toml
[[main]]
name = "About Me"           # Display name in menu
pageRef = "about"           # References about.md file
weight = 10

[[main]]
name = "Posts"
pageRef = "posts"
weight = 20

[[main]]
name = "Tags"
pageRef = "tags"
weight = 30

# Footer menu
[[footer]]
name = "Privacy"
pageRef = "privacy"
weight = 10
```

**Important:** The `pageRef` should match your content file name (without extension). For example:
- `about.md` → `pageRef = "about"`
- `posts/my-post.md` → `pageRef = "posts/my-post"`

The `weight` parameter controls the order (lower numbers appear first), and `name` is what visitors see in the menu.

## Publishing to GitHub Pages

This is where I encountered some challenges. The Blowfish documentation includes a GitHub Actions workflow, but it didn't work as expected for me. Instead, I followed Hugo's official deployment instructions.

### Setting Up GitHub Repository

**1. Create a new repository on GitHub**
- Repository name: `fxd24.github.io` (replace `fxd24` with your GitHub username for user pages) or any name (for project pages)
- Make it public
- Don't initialize with README if using existing local repository

**2. Push your code:**
```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

### GitHub Actions Workflow

Create `.github/workflows/hugo.yaml` (note: use `.yaml` extension):

```yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to GitHub Pages
on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main
  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:
# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write
# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false
# Default to bash
defaults:
  run:
    shell: bash
jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.147.2
      HUGO_ENVIRONMENT: production
      TZ: Europe/Zurich
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb
      - name: Install Dart Sass
        run: sudo snap install dart-sass
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v5
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Cache Restore
        id: cache-restore
        uses: actions/cache/restore@v4
        with:
          path: |
            ${{ runner.temp }}/hugo_cache
          key: hugo-${{ github.run_id }}
          restore-keys:
            hugo-
      - name: Configure Git
        run: git config core.quotepath false
      - name: Build with Hugo
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/" \
            --cacheDir "${{ runner.temp }}/hugo_cache"
      - name: Cache Save
        id: cache-save
        uses: actions/cache/save@v4
        with:
          path: |
            ${{ runner.temp }}/hugo_cache
          key: ${{ steps.cache-restore.outputs.cache-primary-key }}
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public
  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

This is the working configuration I use for my own site. You can find the latest version at: https://github.com/fxd24/fxd24.github.io/blob/main/.github/workflows/hugo.yaml

### Enabling GitHub Pages

**1. Go to your repository settings**
**2. Navigate to Pages section**
**3. Select "GitHub Actions" as the source**
**4. The workflow will automatically trigger on the next push**

## Setting Up Custom Domain

If you want to use a custom domain (like I did with grafdavid.com), here's how:

### DNS Configuration

In your domain registrar, add these DNS records:
```
Type: CNAME
Name: www
Value: fxd24.github.io

Type: A
Name: @
Values: 
    185.199.108.153
    185.199.109.153
    185.199.110.153
    185.199.111.153
```

In your GitHub repository, go to **Settings → Pages** and add your custom domain (e.g., `grafdavid.com`) in the "Custom domain" field.

### GitHub Configuration

**1. In your repository settings, go to Pages**
**2. Add your custom domain in the "Custom domain" field**
**3. Wait for DNS verification (can take up to 24 hours)**
**4. Enable "Enforce HTTPS" once verification is complete**

### Domain Verification

To verify your custom domain on GitHub:
**1. Go to GitHub Settings → Pages**
**2. Add your domain to "Verified domains"**
**3. GitHub will provide a TXT record to add to your DNS**
**4. Add the TXT record and wait for verification**

## Testing Your Site

Before publishing, always test locally:

```bash
# Start development server
hugo server --disableFastRender --noHTTPCache

# Build for production
hugo --minify
```

Visit `http://localhost:1313` to preview your site or any other port that hugo is serving the website onto.


## Conclusion

Setting up a Hugo website with Blowfish and GitHub Pages creates a powerful, fast, and free web presence. While there are a few gotchas (like the empty folder requirement and workflow issues), the end result is a beautiful, performant website that's easy to maintain.

The combination of Hugo's speed, Blowfish's aesthetics, and GitHub Pages' reliability makes this an excellent choice for personal websites, portfolios, and blogs. Once set up, you can focus on creating content while the infrastructure handles itself.

Remember to keep your Hugo version updated and periodically update the Blowfish theme to get the latest features and security improvements.

## Next Steps

Now that your site is live enjoy spreading your expertise and knowledge with the world!


## Friction Log

Feel free to contact me or open an issue on GitHub if you encounter any problems or have suggestions for improvements. I love to minimize friction to deliver better experiences. Here are some common issues I faced and their solutions:

### 1. Blowfish CLI Empty Folder Issue
**Problem:** CLI fails when folder contains existing files
**Solution:** Start with completely empty folder, then add git remote

### 2. GitHub Actions Workflow Failure
**Problem:** Blowfish's provided workflow doesn't work
**Solution:** Use Hugo's official GitHub Actions workflow (provided above)

### 3. Images Not Displaying
**Problem:** Featured images don't show up
**Solution:** Ensure images are named `featured.png` or `featured.jpg` and placed in the same directory as `index.md`

