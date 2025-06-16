# FJSSP Application - Backend Setup Guide

This guide explains how to set up and run the backend components of the Flexible Job Shop Scheduling Problem (FJSSP) application, which consists of a .NET API and Python service. To use the app the frontend part is needed as well, which can be found here [https://github.com/Mihail-Balamatiuc/FJSSP-React-](https://github.com/Mihail-Balamatiuc/FJSSP-React-).

Algorithms used to process the data can be found in pythonService folder in this project or here [https://github.com/Mihail-Balamatiuc/FJSSP](https://github.com/Mihail-Balamatiuc/FJSSP)

## System Requirements

- .NET 9.0 or higher
- Python 3.10 or higher
- Windows, macOS, or Linux operating system

## Setting Up the .NET Backend

1. **Install .NET SDK**
   - Download and install the .NET SDK from [https://dotnet.microsoft.com/download](https://dotnet.microsoft.com/download)
   - Verify installation by running: `dotnet --version`

2. **Clone the Repository**
   - Clone this repository to your local machine

3. **Restore Dependencies**
   - Navigate to the root directory of the application
   - Run: `dotnet restore`

## Setting Up the Python Service

1. **Install Python**
   - Download and install Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Ensure Python is added to your system PATH
   - Verify installation by running: `python --version`

2. **Install Required Python Packages**
   - Navigate to the project directory
   - Run: `pip install pandas plotly matplotlib`
   - In case of a missing library you will be notified anyway, and just run `pip install <library name>`

## Running the Application

1. **Start the .NET Backend**
   - Navigate to the root directory of the application
   - Run: `dotnet run`
   - The API should start at https://localhost:7179 (HTTPS), ensure it's the secure connection

## Datasets

Dataset information and examples can be found here [https://github.com/SchedulingLab/fjsp-instances](https://github.com/SchedulingLab/fjsp-instances)