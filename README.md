# EV Charger Installation Assignment System

An intelligent system for assigning EV charger installation work orders to suitable field teams based on charger compatibility, team availability, workload, and installation duration.

> **Project status:** Academic prototype. The current implementation uses sample/synthetic data in several areas. Validate real team, distance, and operational data before deployment.

## Overview

EV charger installation requests need to be assigned to teams that can handle the charger type and are available on the requested date. This project brings work-order management, team eligibility, assignment recommendations, and reporting into one workflow.

The project includes a Python/Django implementation and a Flutter UI prototype. The Flutter app is a separate frontend and currently uses local sample data; backend/API integration is a planned enhancement.

## Key Features

- **Work-order management:** View, search, filter, and add installation requests.
- **Team management:** Track team locations, charger expertise, installation history, and pending jobs.
- **Eligibility checks:** Consider supported charger type, availability, and workload limits.
- **Smart assignment:** Demonstrate recommendations for eligible teams.
- **Duration prediction:** Machine-learning model for estimating installation duration.
- **Assignment optimization:** Optimization workflow for comparing assignment plans.
- **Reports:** Review assignment outcomes, workload, and distance comparison.
- **Flutter interface:** Dashboard, Work Orders, Teams, Smart Assignment, Reports, and System Information screens.

## Technology Stack

| Area | Technologies |
|---|---|
| Backend / web application | Python, Django |
| Frontend prototype | Flutter, Dart, Material 3 |
| ML | scikit-learn (duration regression model) |
| Optimization | Google OR-Tools / CP-SAT (assignment optimization) |
| Data | Project datasets and sample records |
| Development | VS Code, Git |

## Project Structure

The exact structure may vary by branch/version. A typical layout:

```text
EV_Charger_Assignment/
├── backend/ or Django application files
├── models/
│   └── duration_model.joblib
├── data/                 # datasets (check privacy/licensing before publishing)
├── scripts/              # training and assignment scripts
├── Flutter app/
│   └── lib/
│       ├── main.dart
│       ├── models/
│       ├── data/
│       └── screens/
├── requirements.txt
└── README.md
```

## Assignment Workflow

1. Create or import a work order with charger type, customer location, and requested date.
2. Identify teams that support the requested charger type.
3. Exclude unavailable teams and teams that exceed the workload limit.
4. Estimate installation duration using the trained regression model (Python implementation).
5. Evaluate candidate teams using assignment criteria such as workload and distance.
6. Generate a recommendation or optimized assignment plan.
7. Review the assignment and reports.

## Machine-Learning Model

The duration prediction component was trained on **600 synthetic records** for prototype evaluation.

Reported evaluation from one training run:

| Metric | Value |
|---|---:|
| Selected model | Linear Regression |
| MAE | 0.1606 hours |
| RMSE | 0.1979 hours |
| R² | 0.7999 |

These metrics are based on synthetic data and should not be interpreted as real-world performance. Results may change if the dataset, feature engineering, or train/test split changes.

## Optimization Demonstration

A reported demonstration compared 15 work orders:

| Metric | Demonstration result |
|---|---:|
| Work orders assigned | 15 |
| Team recommendations changed | 6 |
| Original total distance | 126.66 km |
| Optimized total distance | 106.41 km |
| Distance saved | 20.25 km |
| Distance improvement | 15.99% |
| Capacity violations | 0 |

Distances used illustrative locality coordinates and are **not validated road-distance measurements**. Treat these figures as a prototype comparison, not a guaranteed operational saving.

## Installation and Setup

### Python / Django project

1. Clone the repository:

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/EV-Charger-Assignment-System.git
   cd EV-Charger-Assignment-System
   ```

2. Create and activate a virtual environment.

   **Windows PowerShell:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies (if `requirements.txt` is present):

   ```bash
   pip install -r requirements.txt
   ```

4. For a Django application, apply migrations and start the development server:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

   Run these commands from the directory containing `manage.py`.

### Flutter UI

1. Install Flutter and confirm it is available:

   ```bash
   flutter doctor
   ```

2. Open the Flutter app directory (the one containing `pubspec.yaml`), then run:

   ```bash
   flutter pub get
   flutter run
   ```

   To run in Chrome:

   ```bash
   flutter run -d chrome
   ```

## Sample Team Locations

The prototype includes five teams:

| Team | Location |
|---|---|
| T01 | Shivarampally |
| T02 | Attapur |
| T03 | Bachupally |
| T04 | Secunderabad |
| T05 | LB Nagar |

Team experience and installation counts may be sample values. Confirm them before using the system operationally.

## Limitations

- Some datasets and training examples are synthetic.
- Flutter records are currently stored in memory and are not persisted after app restart.
- Flutter dashboard/report values may be demonstration values.
- Flutter assignment duration and distance can be placeholders.
- The Flutter frontend is not yet connected to the Django backend.
- Location/distance calculations require verified coordinates and appropriate routing data.
- The project is an academic prototype, not a production dispatch system.

## Future Enhancements

- Connect Flutter to Django REST APIs.
- Persist work orders, teams, and assignments in a database.
- Add authentication and role-based permissions.
- Use verified geocoding and road-distance/routing data.
- Improve the ML model with real, consented installation records.
- Add live assignment status, notifications, and audit history.
- Expand automated tests and deployment configuration.

## Contributing

For an academic project, contributions can be made through issues or pull requests. Keep credentials, private data, and local environment files out of commits.

## License

Choose and add a license before distributing or reusing this repository. If no license is included, standard copyright restrictions apply.

## Author

**Project:** EV Charger Installation Assignment System  
**Developer:** Add your preferred name and academic details here.
