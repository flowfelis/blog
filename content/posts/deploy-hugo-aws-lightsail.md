+++
title = 'Complete Guide: Deploying Hugo to AWS Lightsail with GitHub Actions'
date = 2026-06-10T22:30:00+02:00
draft = false
+++

Deploying a Hugo static site to an AWS Lightsail VPS gives you complete control and excellent performance. Here is the step-by-step process I used to build and automatically deploy this blog! 😎

## 1. Creating a Lightsail Instance
1. Log into your AWS Lightsail Console.
2. Click **Create instance**.
3. Choose **Linux/Unix** and **OS Only** -> **Ubuntu**.
4. Choose your instance plan (the $5/mo plan is plenty for a static site) and click **Create**.

## 2. Creating a Static IP
By default, AWS changes your IP address if you restart the server. We don't want that. 😁
1. In the Lightsail console, go to the **Networking** tab.
2. Click **Create static IP**.
3. Select your instance from the dropdown to attach it, name the IP, and click **Create**.

## 3. Linking Domain Name to Static IP
1. In the Lightsail console, go to the **Domains & DNS** tab.
2. Click **Create DNS Zone**.
3. Select **Use a domain from another registrar** and enter your domain name, or select **Use a domain that is registered with Amazon Route 53** and select your domain name.
4. Click **Create DNS Zone**
5. Enter to your newly created dns zone and copy the name servers.
6. Go to your Domain Registrar (e.g., GoDaddy, Namecheap, Route53) and paste the name servers copied in prev step.
7. Return to your DNS Zone, and go to **DNS Records** tab.
8. Create an A record with the **value: @**, and **Resolves to: [your-static-ip]**
9. Also, create a CNAME Record to redirect www.yourdomain.com to yourdomain.com

## 4. Setting up Hugo and Nginx
**On your local computer:**
Install Hugo (`brew install hugo`), create your site(https://gohugo.io/getting-started/quick-start/), and push your source code to GitHub. Make sure your `.gitignore` includes the `public/` folder.

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
Make sure the ubuntu user has ownership of this folder:
```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/Projects/blog
```
and also Nginx can access and traverse these directories: 
```bash
chmod +x /home/ubuntu
chmod +x /home/ubuntu/Projects 
chmod +x /home/ubuntu/Projects/blog
```

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
Enable it: 
```bash
sudo ln -s /etc/nginx/sites-available/yourdomain.com /etc/nginx/sites-enabled/
```
and restart Nginx: 
```bash
sudo systemctl restart nginx
```

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
Create `/home/ubuntu/Projects/blog/.github/workflows/deploy.yml` locally:
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
### * Get your SSH Key from AWS Lightsail

  GitHub needs an SSH Private Key so it can securely log into your VPS.

  1. Log into your AWS Lightsail Console.
  2. Go to the Account page (top right menu).
  3. Click on the SSH keys tab.
  4. Download the Default Private Key for the region your VPS is in (it's a  .pem  file).
  5. Open that  .pem  file in a text editor and copy its entire contents (including the  -----BEGIN...  and  -----END...  lines).

  ### * Add Secrets to GitHub

  Now we need to safely store that key and your server's IP address inside GitHub.

  1. Go to your repository on GitHub.com.
  2. Click Settings (at the top of the repo).
  3. On the left sidebar, scroll down to Secrets and variables -> Actions.
  4. Click the green New repository secret button.

  You need to create three secrets exactly like this:

  1. Name:  VPS_HOST
  Secret: Your AWS Lightsail IP address (e.g.,  12.34.56.78 )
  
  2. Name:  VPS_USERNAME
  Secret:  ubuntu
  
  3. Name:  VPS_SSH_KEY
  Secret: Paste the entire contents of the  .pem  file you copied earlier.

Now, every time you `git push` from your computer, your site updates automatically! 😎

You don't even need to ssh to your VPS!
