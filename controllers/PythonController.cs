using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("/[controller]")]
public class PythonServiceController : ControllerBase
{

    private readonly IWebHostEnvironment _env;

    // Constructor: Injects IWebHostEnvironment
    public PythonServiceController(IWebHostEnvironment env)
    {
        _env = env;
    }

    // PUT endpoint to handle file upload and script execution
    [HttpPut("start")]
    public async Task<IActionResult> StartProcess(IFormFile file)
    {
        try
        {
            // Check if a file was uploaded
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded");
            }

            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "dataset_github.txt");

            // Overwrite the existing with the uploaded file's content, but doesn't replace the file so it's name stays the same
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(stream); // Copy uploaded file to dataset_github.txt
            }

            // Define the Python script path
            string scriptPath = Path.Combine(_env.ContentRootPath, "pythonService", "main_schedule.py");


            if (!System.IO.File.Exists(scriptPath))
            {
                return StatusCode(500, $"Script not found at: {scriptPath}");
            }

            // Configure the process to run the Python script
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "python", // Assumes 'python' is in your system PATH
                    // Here we add the <"> symbol at the end and beggining of the path to handle path spaces.
                    // ex: <C:\My Things> has spaces and will not be accesed correctly, so we need to make <"C:\My Things"> and now it's taken
                    // like a whole string, it's a common practice when working with paths
                    Arguments = "\"" + scriptPath + "\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            // Start the Python script and read its output/errors
            process.Start();
            string output = await process.StandardOutput.ReadToEndAsync();
            string error = await process.StandardError.ReadToEndAsync();
            // Waits for the script to run, for long running scripts make it async
            await process.WaitForExitAsync();

            // Check if the script didn't fail
            if (process.ExitCode != 0)
            {
                return StatusCode(500, $"Error: {error}");
            }
            // Writes the output in the needed file
            string outputPath = Path.Combine(_env.ContentRootPath, "pythonService", "scriptOutput.txt");
            await System.IO.File.WriteAllTextAsync(outputPath, output);

            return Ok(output);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Exception: {ex.Message}");
        }
    }

    // PUT endpoint to handle file upload and script execution
    [HttpPut("compare")]
    public async Task<IActionResult> Compare(IFormFile file)
    {
        try
        {
            // Check if a file was uploaded
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded");
            }

            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "dataset_github_compare.txt");

            // Overwrite the existing input with the uploaded file's content, but doesn't replace the file so it's name stays the same
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            // Define the Python script path
            string scriptPath = Path.Combine(_env.ContentRootPath, "pythonService", "main_compare.py");


            if (!System.IO.File.Exists(scriptPath))
            {
                return StatusCode(500, $"Script not found at: {scriptPath}");
            }

            // Configure the process to run the Python script
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "python", // Assumes 'python' is in your system PATH
                    // Here we add the <"> symbol at the end and beggining of the path to handle path spaces.
                    // ex: <C:\My Things> has spaces and will not be accesed correctly, so we need to make <"C:\My Things"> and now it's taken
                    // like a whole string, it's a common practice when working with paths
                    Arguments = "\"" + scriptPath + "\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            // Start the Python script and read its output/errors
            process.Start();
            string output = await process.StandardOutput.ReadToEndAsync();
            string error = await process.StandardError.ReadToEndAsync();
            // Waits for the script to run, for long running scripts make it async
            await process.WaitForExitAsync();

            // Check if the script didn't fail
            if (process.ExitCode != 0)
            {
                return StatusCode(500, $"Error: {error}");
            }

            return Ok(output);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Exception: {ex.Message}");
        }
    }

    // GET endpoint to retrieve the current python configuration
    [HttpGet("getConfig")]
    // We make it synchronous because it's a simple operation
    public async Task<IActionResult> GetConfig()
    {
        try
        {
            string configPath = Path.Combine(_env.ContentRootPath, "pythonService", "config.json");
            // Check if the path exists
            if (!System.IO.File.Exists(configPath))
            {
                return NotFound("Configuration file not found");
            }
            string configContent = await System.IO.File.ReadAllTextAsync(configPath);
            return Ok(configContent);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error reading config: {ex.Message}");
        }
    }

    // PUT endpoint to save the updated configuration
    [HttpPut("saveConfig")]
    public async Task<IActionResult> SaveConfig([FromBody] JsonElement jsonConfig)
    {
        try
        {
            // Grab the raw JSON string exactly how it was sent:
            string rawJson = jsonConfig.GetRawText();

            if (string.IsNullOrWhiteSpace(rawJson))
            {
                return BadRequest("Empty request body.");
            }

            // Write the rawJson to the file
            string configPath = Path.Combine(_env.ContentRootPath, "pythonService", "config.json");

            // here’s the sync version of writing to the file:
            await System.IO.File.WriteAllTextAsync(configPath, rawJson);

            return Ok("Configuration saved successfully.");
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error saving file: {ex.Message}");
        }
    }

    // GET endpoint to download the schedule text file
    [HttpGet("getSchedule")]
    public async Task<IActionResult> GetSchedule()
    {
        try
        {
            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "scriptOutput.txt");
            if (!System.IO.File.Exists(filePath))
            {
                return NotFound("Schedule file not found");
            }
            return PhysicalFile(filePath, "text/plain", "schedule.txt");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error serving schedule: {ex.Message}");
            return StatusCode(500, "An error occurred while retrieving the schedule");
        }
    }

    // Get endpoint for the schedule algorithms
    [HttpGet("getScheduleAlgorithms")]
    public async Task<IActionResult> GetScheduleAlgorithms()
    {
        try
        {
            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "schedule_algorithms.txt");

            // Check if the file exists
            if (!System.IO.File.Exists(filePath))
            {
                return NotFound("Schedule algorithms file not found");
            }
            string content = await System.IO.File.ReadAllTextAsync(filePath);

            // Split the text into strings
            string[] algorithms = content.Split([' ', '\r', '\n', '\t'],
            StringSplitOptions.RemoveEmptyEntries); // Remove empty strings from the array if there are any

            return Ok(algorithms);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error serving schedule algorithms file: {ex.Message}");
            return StatusCode(500, "An error occurred while retrieving the schedule algorithms");
        }
    }

    // PUT request to update the schedule algorithms(algorithm names)
    [HttpPut("saveScheduleAlgorithms")]
    public async Task<IActionResult> SaveScheduleAlgorithms([FromBody] List<string> algorithms)
    {
        try
        {
            // Only validate that algorithms is not null
            if (algorithms == null)
            {
                return BadRequest("Algorithm list cannot be null");
            }

            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "schedule_algorithms.txt");

            // Check if the file exists
            if (!System.IO.File.Exists(filePath))
            {
                return NotFound("Schedule algorithms file not found");
            }

            // If the list is empty, this will produce an empty string
            string algorithmContent = string.Join(" ", algorithms);
            await System.IO.File.WriteAllTextAsync(filePath, algorithmContent);

            return Ok("The schedule algorithm list was succesfully updated");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving schedule algorithms file: {ex.Message}");
            return StatusCode(500, "An error occurred while saving the schedule algorithms");
        }
    }

    // Get endpoint for comparing algorithms
    [HttpGet("getCompareAlgorithms")]
    public async Task<IActionResult> GetCompareAlgorithms()
    {
        try
        {
            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "compare_algorithms.txt");

            // Check if the file exists
            if (!System.IO.File.Exists(filePath))
            {
                return NotFound("Compare algorithms file not found");
            }
            string content = await System.IO.File.ReadAllTextAsync(filePath);

            // Split the text into strings
            string[] algorithms = content.Split([' ', '\r', '\n', '\t'],
            StringSplitOptions.RemoveEmptyEntries); // Remove empty strings from the array if there are any

            return Ok(algorithms);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error serving compare algorithms file: {ex.Message}");
            return StatusCode(500, "An error occurred while retrieving the compare algorithms");
        }
    }

    // PUT request to update the schedule algorithms(algorithm names)
    [HttpPut("saveCompareAlgorithms")]
    public async Task<IActionResult> SaveCompareAlgorithms([FromBody] List<string> algorithms)
    {
        try
        {
            // Only validate that algorithms is not null
            if (algorithms == null)
            {
                return BadRequest("Algorithm list cannot be null");
            }

            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "compare_algorithms.txt");

            // Check if the file exists
            if (!System.IO.File.Exists(filePath))
            {
                return NotFound("Compare algorithms file not found");
            }

            // If the list is empty, this will produce an empty string
            string algorithmContent = string.Join(" ", algorithms);
            await System.IO.File.WriteAllTextAsync(filePath, algorithmContent);

            return Ok("The compare algorithm list was succesfully updated");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving compare algorithms file: {ex.Message}");
            return StatusCode(500, "An error occurred while saving the compare algorithms");
        }
    }
}