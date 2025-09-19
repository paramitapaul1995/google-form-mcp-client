
#!/usr/bin/env python3
"""
OpenAI client that connects to the Google Forms MCP server using FastMCP.
This script:
1. Connects to the Google Forms MCP server using FastMCP client
2. Fetches available tools and converts them to OpenAI function format
3. Uses OpenAI to process user requests and execute Google Forms operations
Usage:
  python openai_google_forms_client.py "Create a survey about customer satisfaction"
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI

class MCPClient:
    """FastMCP-based client that communicates with the Google Forms server."""
    
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.client = None
    
    async def start(self):
        """Start the MCP client connection."""
        self.client = Client(self.server_path)
        await self.client.__aenter__()
    
    async def stop(self):
        """Stop the MCP client connection."""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the MCP server."""
        if not self.client:
            raise RuntimeError("MCP client not started")
        return await self.client.list_tools()
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.client:
            raise RuntimeError("MCP client not started")
        return await self.client.call_tool(name, arguments)



def build_openai_tools_schema(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert MCP tools to OpenAI function calling format."""
    schemas = []
    for tool in tools:
        # Handle both Tool objects and dictionaries
        if hasattr(tool, 'name'):
            # Tool object (Pydantic model)
            name = tool.name
            description = tool.description or ""
            input_schema = tool.inputSchema or {"type": "object", "properties": {}}
        else:
            # Dictionary
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})
        
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        })
    return schemas

def tool_result_to_text(result: Any) -> str:
    """Convert MCP tool result to text for OpenAI."""
    try:
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts)
        return str(result)
    except Exception:
        return str(result)

async def main():
    """Main function to run the OpenAI + Google Forms integration."""
    load_dotenv()
    
    # Check for required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment/.env")
    
    # Path to the built MCP server
    server_path = os.getenv("GOOGLE_FORMS_MCP_PATH", r"C:\Users\PARAMITA PAUL\VV_Courses\googleforms\google-forms-mcp\build\index.js")
    if not server_path or not os.path.exists(server_path):
        raise RuntimeError(f"Google Forms MCP server not found at {server_path}")
    
    # Get user input
    # Get user input
    print("=== Google Forms Creator ===")
    print("Enter your form details:")
    print("Format: 'Title: [Your Title] | Questions: [Question 1] | [Question 2] | ...'")
    print("Example: 'Title: Customer Feedback | Questions: required What are your comments? | How satisfied are you? (Very Satisfied, Satisfied, Neutral, Dissatisfied, Very Dissatisfied)'")
    print("\nTo mark questions as REQUIRED, prefix them with 'required':")
    print("Example: 'Title: Survey | Questions: required How would you rate me? (Good, Bad) | What do you think? | required What should I improve?'")
    print("=" * 80)
    
    user_input = input("Enter your form request: ").strip()
    if not user_input:
        print("Error: Please provide form details!")
        return
    
    print(f"\nProcessing: {user_input}")
    print("=" * 50)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Start MCP server and get tools
    mcp_client = MCPClient(server_path)
    try:
        await mcp_client.start()
        tools = await mcp_client.list_tools()
        openai_tools = build_openai_tools_schema(tools)
        
        # Prepare messages for OpenAI
        messages = [
            {
    "role": "system", 
    "content": "You are a helpful assistant that creates Google Forms. IMPORTANT: When creating forms, you can ONLY set the title during creation - do NOT include a description parameter. After creating a form, you MUST immediately add ALL the requested questions using add_text_question and add_multiple_choice_question tools in the EXACT ORDER they appear in the user input. Do not reorder questions - maintain the sequence provided. Do not stop after creating the form - continue until all questions are added. Parse the user input to extract the title and questions. For multiple choice questions, if options are provided in parentheses, use them. Otherwise, create reasonable default options. REQUIRED QUESTIONS: If a question is prefixed with 'required' (e.g., 'required How would you rate me...'), set required: true. If no 'required' prefix is present, set required: false. Always provide the final form URL when complete."
},
            {"role": "user", "content": user_input},
        ]
        
        # First OpenAI call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.2,
        )
        
        choice = response.choices[0]
        msg = choice.message
        
        if msg.content:
            print("Assistant:", msg.content)
        
        # Execute tool calls if any
        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls
            })
            
            # Keep executing tools until AI is done
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Execute all tool calls in this round
                tool_calls_made = False
                for tc in msg.tool_calls:
                    if tc.type == "function":
                        name = tc.function.name
                        arguments = tc.function.arguments or "{}"
                        try:
                            args = json.loads(arguments)
                        except Exception as e:
                            print(f"Error parsing arguments for {name}: {e}")
                            args = {}
                        
                        print(f"\n[Executing tool] {name}({args})")
                        try:
                            result = await mcp_client.call_tool(name, args)
                            result_text = tool_result_to_text(result)
                            print("Tool result:", result_text)
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            })
                            tool_calls_made = True
                        except Exception as e:
                            print(f"Error executing tool {name}: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Error: {str(e)}",
                            })
                            tool_calls_made = True
                
                if not tool_calls_made:
                    break
                    
                # Ask AI to continue adding questions
                continue_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages + [{"role": "user", "content": "Continue adding the remaining questions. Don't ask for confirmation - just add them immediately."}],
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0.2,
                )
                
                choice = continue_response.choices[0]
                msg = choice.message
                
                if msg.content:
                    print("Assistant:", msg.content)
                
                # If no more tool calls, we're done
                if not getattr(msg, "tool_calls", None):
                    break
                    
                # Add the assistant's response to messages
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls
                })
            
            print(f"\n--- Completed after {iteration} iterations ---")
            
            # Final response after all tool execution
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
            )
            print("\nFinal response:", final_response.choices[0].message.content)
    
    finally:
        await mcp_client.stop()

if __name__ == "__main__":
    asyncio.run(main())
