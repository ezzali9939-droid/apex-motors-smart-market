# APEX Motors

## Requirements
- Python 3.9+

## Install
```bash
pip install -r requirements.txt
```

## Required Environment Variables
- `GEMINI_API_KEY` (for vehicle vision processing)

Create a `.env` file from the example:
```bash
cp .env.example .env
```
And add your secret key.

## Local run command
To run the server locally:
```bash
python api/index.py
```
*(Or use Flask development server if preferred).*

## Testing command
Run tests with:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Vercel Deployment Notes
This project is configured for Vercel Serverless deployment using the `vercel.json` configuration file. Ensure environment variables are set in your Vercel project settings.