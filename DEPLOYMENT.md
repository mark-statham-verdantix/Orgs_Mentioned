# Streamlit Community Deployment Guide

## 🚀 Deploy Your NER App Securely

This guide shows you how to deploy your organization extraction app to Streamlit Community while keeping your database credentials secure.

## Prerequisites

✅ Your code is already configured to use Streamlit secrets  
✅ Your `.gitignore` protects sensitive files  
✅ You have a GitHub account  

## Step-by-Step Deployment

### 1. **Push to GitHub**

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Add NER Streamlit app with secrets support"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### 2. **Deploy to Streamlit Community**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set these deployment settings:
   - **Repository**: `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch**: `main`
   - **Main file path**: `ner_streamlit.py`
   - **App URL**: Choose a memorable name like `your-org-ner-app`

### 3. **Add Database Secrets (CRITICAL)**

After deployment, immediately add your database credentials:

1. Go to your deployed app dashboard
2. Click **Settings** → **Secrets**
3. Add this configuration (replace with your actual values):

```toml
[database]
POSTGRES_HOST = "your-actual-database-host.com"
POSTGRES_PORT = "5432"
POSTGRES_DATABASE = "your-database-name"
POSTGRES_USER = "your-read-only-username"
POSTGRES_PASSWORD = "your-secure-password"
```

4. Click **Save**
5. Your app will automatically restart with the new secrets

## 🔒 Security Best Practices

### Database Security
- [ ] Create a **read-only** database user specifically for this app
- [ ] Grant only `SELECT` permissions on the `verdantix.org` table
- [ ] Use a strong, unique password
- [ ] Consider IP whitelisting if your database supports it

### Access Control
- [ ] Share the app URL only with authorized colleagues
- [ ] Monitor your database logs for unusual access patterns
- [ ] Consider setting up database connection limits

### Example Read-Only Database User Setup

```sql
-- Create read-only user
CREATE USER 'streamlit_readonly'@'%' IDENTIFIED BY 'secure_random_password';

-- Grant only SELECT permission on specific table
GRANT SELECT ON verdantix.org TO 'streamlit_readonly'@'%';

-- Apply changes
FLUSH PRIVILEGES;
```

## 🎯 Testing Your Deployment

1. Visit your deployed app URL
2. Upload a test document (try Test Files/Test1/P5 Text.docx)
3. Verify organizations are extracted correctly
4. Test export functionality
5. Share with a colleague to confirm access works

## 🔧 Troubleshooting

### "Database configuration missing" Error
- Check that secrets are properly formatted in TOML
- Verify no extra spaces in secret keys/values
- Ensure your database accepts connections from external IPs

### Connection Timeout
- Your database might not accept external connections
- Check firewall settings
- Verify database host and port are correct

### App Won't Start
- Check the app logs in Streamlit Community dashboard
- Verify all required packages are in `requirements.txt`
- Ensure your Python code has no syntax errors

## 📞 Support

If you encounter issues:
1. Check the Streamlit Community app logs
2. Test database connectivity locally first
3. Verify secrets formatting matches the example exactly

## 🔄 Updating Your App

To update your deployed app:
1. Push changes to your GitHub repository
2. Streamlit Community will automatically redeploy
3. Secrets persist across deployments

---

**Your app is now securely deployed and ready to share with colleagues!** 🎉