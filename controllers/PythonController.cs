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

    // POST endpoint to handle file upload and script execution
    [HttpPost("start")]
    public async Task<IActionResult> StartProcess(IFormFile file)
    {
        try
        {
            // Check if a file was uploaded
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded");
            }

            // Define the fixed file path (input.txt) in the pythonService folder
            string filePath = Path.Combine(_env.ContentRootPath, "pythonService", "dataset_github.txt");

            // Overwrite the existing input.txt with the uploaded file's content, but doesn't replace the file so it's name stays the same
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(stream); // Copy uploaded file to dataset_github.txt
            }

            // Define the Python script path
            string scriptPath = Path.Combine(_env.ContentRootPath, "pythonService", "main2.py");


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
    public IActionResult GetConfig()
    {
        try
        {
            string configPath = Path.Combine(_env.ContentRootPath, "pythonService", "config.json");
            // Check if the path exists
            if (!System.IO.File.Exists(configPath))
            {
                return NotFound("Configuration file not found");
            }
            string configContent = System.IO.File.ReadAllText(configPath);
            return Ok(configContent);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error reading config: {ex.Message}");
        }
    }

    // POST endpoint to save the updated configuration
    [HttpPost("saveConfig")]
    public IActionResult SaveConfig([FromBody] JsonElement jsonConfig)
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
            System.IO.File.WriteAllText(configPath, rawJson);

            return Ok("Configuration saved successfully.");
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error saving file: {ex.Message}");
        }
    }

}