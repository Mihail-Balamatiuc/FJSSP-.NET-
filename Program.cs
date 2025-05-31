var builder = WebApplication.CreateBuilder(args);
// Here before build we add the services

// Add CORS services
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowReactApp",
        builder => builder.WithOrigins("http://localhost:5173")
                          .AllowAnyMethod()
                          .AllowAnyHeader());
});

// Add controllers for handling HTTP requests
builder.Services.AddControllers();

// Add OpenAPI/Swagger services for API
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
// Use HTTPS redirection to ensure secure connections
app.UseHttpsRedirection();

// Use routing to enable endpoint mapping
app.UseRouting();

// Enable CORS
app.UseCors("AllowReactApp");

// Tells the app to route requests to controllers
app.MapControllers();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/openapi/v1.json", "v1");
    });
}

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.Run();