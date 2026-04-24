// Counter module in Verilog
module counter (
  input wire clk,
  output reg [7:0] count // 8-bit counter
);

always @(posedge clk) begin
    count <= count + 1;
end

endmodule