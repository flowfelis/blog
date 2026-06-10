+++
title = 'Complete Guide: Deploying Hugo to AWS Lightsail with GitHub Actions'
date = 2026-06-10T22:30:00+02:00
draft = false
+++

Deploying a Hugo static site to an AWS Lightsail VPS gives you complete control and excellent performance. Here is the step-by-step process we used to build and automatically deploy this blog!

## 1. Creating a Lightsail Instance
1. Log into your AWS Lightsail Console.
2. Click **Create instance**.
3. Choose **Linux/Unix** and **OS Only** -> **Ubuntu**.
4. Choose your instance plan (the $3.50/mo plan is plenty for a static site) and click **Create**.

## 2. Creating a Static IP
By default, AWS changes your IP address if you restart the server.
1. In the Lightsail console, go to the **Networking** tab.
2. Click **Create static IP**.
3. Select your instance from the dropdown to attach it, name the IP, and click **Create**.

## 3. Linking Domain Name to Static IP
1. Go to your Domain Registrar (e.g., GoDaddy, Namecheap, Route53).
2. Go to your DNS management settings.
3. Create a new **A Record**. Set the host/name to `@` and the value to your new Lightsail Static IP.

## 4. Setting up Hugo and Nginx
**On your Local Mac:**
Install Hugo (`brew install hugo`), create your site, and push your source code to GitHub. Make sure your `.gitignore` includes the `public/` folder.

**On your VPS:**
SSH into your server and install the requirements:
```bash
sudo apt update
sudo apt install git nginx -y
sudo snap install hugo
```
Clone your GitHub repo and build the site:
```bash
git clone https://github.com/yourusername/blog.git /home/ubuntu/Projects/blog
cd /home/ubuntu/Projects/blog
hugo
```
*(Make sure the ubuntu user has ownership of this folder: `sudo chown -R ubuntu:ubuntu /home/ubuntu/Projects/blog` and that Nginx can access your home directory: `chmod +x /home/ubuntu`).*

Configure Nginx (`sudo nano /etc/nginx/sites-available/yourdomain.com`):
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    root /home/ubuntu/Projects/blog/public;
    index index.html;
    location / {
        try_files $uri $uri/ =404;
    }
}
```
Enable it: `sudo ln -s /etc/nginx/sites-available/yourdomain.com /etc/nginx/sites-enabled/` and restart Nginx (`sudo systemctl restart nginx`).

## 5. Setting up Let's Encrypt (SSL)
Secure your site with a free SSL certificate:
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

## 6. Firewall Setting for HTTPS (Port 443)
If you try to load your site now, it will timeout. You must open port 443!
1. In Lightsail, click your instance and go to the **Networking** tab.
2. Under **IPv4 Firewall**, click **+ Add rule**.
3. Choose **HTTPS** (TCP, port 443) and save.

## 7. Verifying Site Visibility
At this point, you can navigate to `https://yourdomain.com` in your browser. You should see your secure Hugo blog live on the web!

## 8. Automating Deployments with GitHub Actions
To avoid logging into the server every time you write a post, set up GitHub Actions.
Create `.github/workflows/deploy.yml` locally:
```yaml
name: Deploy Hugo Site
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USERNAME }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            cd /home/ubuntu/Projects/blog
            git pull origin main
            hugo
```
Finally, add your `VPS_HOST` (static IP), `VPS_USERNAME` (ubuntu), and `VPS_SSH_KEY` (Lightsail default private key) into your GitHub repository's **Secrets**. 

Now, every time you `git push` from your Mac, your site updates automatically!
