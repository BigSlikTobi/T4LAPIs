# GitHub Actions Setup Guide

## 🚀 Sprint 3 Epic 1 Task 2: Automated Personalized Summary Generation

This guide covers setting up the GitHub Actions workflow for automated personalized summary generation.

## 📋 Prerequisites

1. **Repository Secrets**: The following secrets must be configured in your GitHub repository
2. **Database Access**: Working Supabase database with user preferences
3. **LLM API Keys**: Valid Gemini and DeepSeek API keys

## 🔐 Required GitHub Secrets

Navigate to your repository → Settings → Secrets and variables → Actions, then add these secrets:

### Database Secrets
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### LLM API Secrets
```
GEMINI_API_KEY=your_gemini_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 📅 Workflow Schedule

The workflow runs automatically:
- **Every hour** during active hours (6 AM - 11 PM UTC)
- **Covers peak NFL activity** across US time zones
- **Manual trigger** available for testing

## 🧪 Testing the Workflow

### 1. Local Validation
```bash
# Run the validation test
python tests/test_github_actions_workflow.py
```

### 2. Manual GitHub Actions Test
1. Go to Actions tab in your GitHub repository
2. Select "🤖 Personalized Summary Generation"
3. Click "Run workflow"
4. Set test parameters:
   - Force generation: `false` (normal operation)
   - Lookback hours: `1` (short test period)

### 3. Monitor First Run
- Check workflow logs for any issues
- Verify summaries are generated in database
- Confirm no API rate limit issues

## 📊 Workflow Features

### Smart Scheduling
- **Hourly execution**: `0 6-23 * * *` (6 AM to 11 PM UTC)
- **Timeout protection**: 30-minute maximum runtime
- **Manual override**: Available for testing and debugging

### LLM Integration
- **Primary**: Gemini 1.5 Flash for superior content quality
- **Fallback**: DeepSeek Chat for reliability
- **Connection testing**: Validates both APIs before execution

### Error Handling
- **Graceful failures**: Detailed error reporting
- **Recovery guidance**: Step-by-step troubleshooting
- **Notification**: Clear success/failure indicators

### Performance Monitoring
- **Execution tracking**: Runtime and generation statistics
- **Summary counting**: Tracks output generation
- **Resource monitoring**: Prevents runaway processes

## 🔧 Workflow Configuration

### Environment Variables Set by Workflow
```yaml
SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
LOG_LEVEL: INFO
```

### Customizable Parameters (Manual Runs)
- **Force generation**: Generate summaries even without new content
- **Lookback hours**: Adjust time window for content gathering

## 📈 Expected Output

### Successful Run Log
```
✅ Dependencies installed successfully
✅ Environment variables configured
✅ Gemini connection successful
✅ DeepSeek connection successful
🚀 Starting personalized summary generation...
✅ Personalized summary generation completed
📝 Generated 12 new summaries
```

### Database Impact
- New records in `generated_updates` table
- Personalized summaries for users with preferences
- Timestamped entries for tracking

## 🚨 Troubleshooting

### Common Issues

#### 1. LLM Connection Failures
```
❌ Gemini connection failed: API key invalid
```
**Solution**: Verify `GEMINI_API_KEY` secret is correct

#### 2. Database Connection Issues
```
❌ Database connection failed: Invalid API key
```
**Solution**: Check `SUPABASE_URL` and `SUPABASE_KEY` secrets

#### 3. No Summaries Generated
```
✅ Workflow completed but 0 summaries generated
```
**Possible causes**:
- No users with preferences
- No new content in lookback period
- Database schema issues

### Debug Steps

1. **Check GitHub Secrets**
   - Verify all 4 required secrets are set
   - Ensure no trailing spaces or extra characters

2. **Run Manual Test**
   - Use workflow_dispatch with short lookback period
   - Monitor logs step-by-step

3. **Validate Database**
   - Confirm users exist in `users` table
   - Verify preferences in `user_preferences` table
   - Check article content in last 24 hours

4. **Test LLM APIs**
   - Run local validation: `python tests/test_github_actions_workflow.py`
   - Check API key validity and rate limits

## 🎯 Success Metrics

### Key Performance Indicators
- **Reliability**: >95% successful workflow executions
- **Coverage**: Summaries generated for all active users
- **Performance**: <10 minutes average execution time
- **Quality**: Engaging, well-formatted summary content

### Monitoring Commands
```bash
# Check recent workflow runs
gh run list --workflow="personal_summary_generation.yml"

# View specific run logs
gh run view [run-id] --log

# Check generated summaries count
python -c "
from src.core.db.database_init import get_supabase_client
client = get_supabase_client()
response = client.table('generated_updates').select('update_id').execute()
print(f'Total summaries: {len(response.data)}')
"
```

## 🎉 Deployment Checklist

- [ ] GitHub secrets configured (4 required)
- [ ] Workflow file committed to `.github/workflows/`
- [ ] Local validation test passes
- [ ] Manual workflow test successful
- [ ] Database contains users with preferences
- [ ] LLM APIs have sufficient quotas
- [ ] Monitoring setup for workflow notifications

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Supabase API Reference](https://supabase.com/docs/reference/api)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [DeepSeek API Documentation](https://platform.deepseek.com/docs)
