
## Tech Stack

- React 18.2.0
- Material-UI (MUI) 5.13.0
- Chart.js for data visualization
- React Router for navigation
- Axios for API communication

## Project Structure

```
frontend/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── RisksAndChallenges.jsx    # Risk analysis component
│   │   └── ... 
│   ├── pages/            # Main page components
│   │   ├── DocumentAnalysisPage.jsx   # Document processing interface
│   │   ├── HomePage.jsx               # Landing page
│   │   ├── StockRecommendationsPage.jsx  # Stock analysis
│   │   └── TradeSimulationPage.jsx    # Trading simulator
│   ├── services/         # API and utility services
│   │   └── api.js       # Backend API integration
│   ├── App.jsx          # Main application component
│   └── index.jsx        # Application entry point
└── public/              # Static assets
```


## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Build for production:
```bash
npm run build
```

## Configuration

The application uses environment variables for configuration:
- `REACT_APP_API_URL`: Backend API endpoint
- `REACT_APP_ENV`: Environment (development/production)

Create a `.env` file in the root directory to set these variables.

## API Integration

The frontend communicates with the backend through RESTful APIs:
- `/api/documents/process`: Document analysis endpoint
- `/api/market-data`: Market data retrieval
- `/api/recommendations`: Stock recommendations
- `/api/simulation`: Trade simulation endpoints

## Styling

- Material-UI (MUI) components for consistent design
- Responsive layout for all screen sizes
- Custom theme configuration in `src/theme.js`
- CSS-in-JS using MUI's styling solution

## Best Practices

- Component-based architecture
- Proper error handling and loading states
- Responsive design principles
- Clean and maintainable code structure
- Comprehensive documentation

