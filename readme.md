# Horse Racing Analytics Dashboard

## Overview
A comprehensive analytics dashboard exploring horse racing and sports betting through data-driven insights. This project combines academic research, statistical analysis, and betting insights to provide deeper understanding of horse racing dynamics.

## Features

### 1. Literature Review
- Comprehensive overview of academic research in horse racing analytics
- Key findings from sports betting literature
- Historical perspectives on betting strategies

### 2. Betting Calculator & Payout Analysis
- Interactive payout calculator
- Detailed explanation of betting odds and structures
- Risk assessment tools

### 3. Major Racing Events
- Calendar of significant horse racing events worldwide
- Historical data and trends for major races
- Prize pool analysis

### 4. Data Analysis
- Statistical analysis of racing outcomes
- Track condition impact studies
- Jockey and trainer performance metrics
- Historical odds analysis

### 5. Predictive Modeling
- Machine learning approaches to race prediction
- Feature importance analysis
- Model performance metrics
- Backtesting results

### 6. Personal Portfolio
- Professional background
- Working principles and methodology
- CV and relevant experience

## Setup

### Prerequisites
- Python 3.8 or higher
- Poetry for dependency management

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the Streamlit app
poetry run streamlit run app.py
```

### Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=your_database_url
API_KEY=your_api_key
```

## Development

### Project Structure
```
├── app.py              # Main Streamlit application
├── data/              # Data files and datasets
├── models/            # Machine learning models
├── analysis/          # Analysis scripts and notebooks
├── tests/             # Test files
└── utils/             # Utility functions
```

### Development Tools

#### Project Context Generator
A utility script that generates a comprehensive project overview.

```bash
# Install development dependencies
poetry install --with dev

# Run the script
poetry run python tools/context_generator.py
```

The script will automatically include:
- `.gitignore`
- `pyproject.toml`
- `README.md`
- `app.py`
- `.env`

### Testing
```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov
```

### Code Quality
```bash
# Run linting
poetry run flake8

# Run type checking
poetry run mypy .
```

## Deployment

### Local Development
```bash
poetry run streamlit run app.py
```

### Production Deployment
[Add specific deployment instructions based on your hosting platform]

## Data Sources
- Historical race records from [source]
- Betting odds data from [source]
- Track condition reports from [source]

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
[Your chosen license]

## Contact
[Your contact information]

## Acknowledgments
- [Any acknowledgments]
- [References to key research papers]
- [Credits to data sources]